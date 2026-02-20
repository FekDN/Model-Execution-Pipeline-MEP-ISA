# MEP_interpreter.py
#
# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# Этот файл содержит MEPInterpreter — виртуальную машину, которая исполняет
# бинарный байт-код, сгенерированный MEPCompiler'ом. Она управляет
# контекстом исполнения, ресурсами и потоком выполнения плана.

import struct
import sys
import numpy as np
from typing import List, Dict, Any
import json

# Предполагается, что эти зависимости доступны
from NAC_run import NacRuntime
from NAC_kernels import softmax

# Дополнительные зависимости для новых инструкций
from PIL import Image
import io

class MEPInterpreter:
    """
    Интерпретатор для Model Execution Pipeline (MEP) ISA v1.0.
    """
    def __init__(self, execution_plan: bytes, constants_pool: Dict[int, Any]):
        print("--- Инициализация MEP интерпретатора ---")
        self.plan = execution_plan
        self.constants = constants_pool
        self.ip = 0  # Instruction Pointer
        self.context: List[Any] = [None] * 256
        self.resources: Dict[int, Any] = {}
        self.return_value: Any = None
        self.running = False
        # Стек для отслеживания активных циклов во время исполнения
        self.loop_stack: List[Dict[str, Any]] = []

        # Словарь длин инструкций для корректного пропуска (дизассемблирования)
        # Ключ: опкод, Значение: длина в байтах (или -1 для переменной)
        self._instr_lengths = {
            0x02: 4, 0x04: 3, 0x10: 3, 0x11: 4, 0x12: 4, 0x1F: 2,
            0x20: 3, 0x21: 3, 0x22: 4, 0x2A: -1, # переменная
            0x30: 4, 0x38: -1, 0x39: -1, 0x3A: -1, 0x3B: 3,
            0x59: 2, 0x5F: 3,
            0x60: 3, 0x61: 4, 0x62: 3, 0x68: 4,
            0x80: -1,
            0xA0: 2, 0xA1: 3, 0xA8: 4, 0xA9: 4,
            0xE0: 3, # без ...params
            0xF0: 4,
            0xFE: -1, 0xFF: 1
        }

        self.handlers = {
            # --- 0x00-0x0F: Источники данных ---
            0x02: self._handle_src_user_prompt,
            0x04: self._handle_src_constant,
            # --- 0x10-0x1F: Управление ресурсами ---
            0x10: self._handle_res_load_model,
            0x11: self._handle_res_load_datafile,
            0x12: self._handle_res_load_extern,
            0x1F: self._handle_res_unload,
            # --- 0x20-0x2F: Препроцессинг ---
            0x20: self._handle_preproc_encode,
            0x21: self._handle_preproc_decode,
            0x22: self._handle_preproc_get_id,
            0x2A: self._handle_string_format,
            # --- 0x30-0x4F: Обработка тензоров ---
            0x30: self._handle_tensor_create,
            0x38: self._handle_tensor_manipulate,
            0x39: self._handle_tensor_combine,
            0x3A: self._handle_tensor_info,
            0x3B: self._handle_tensor_extract,
            # --- 0x50-0x5F: Системные вызовы ---
            0x59: self._handle_sys_copy,
            0x5F: self._handle_sys_debug_print,
            # --- 0x60-0x7F: Постобработка и Логика ---
            0x60: self._handle_math_unary,
            0x61: self._handle_math_binary,
            0x62: self._handle_math_aggregate,
            0x68: self._handle_logic_compare,
            # --- 0x80-0x8F: Выполнение модели ---
            0x80: self._handle_model_run_static,
            # --- 0xA0-0xAF: Управление потоком ---
            0xA0: self._handle_flow_loop_start,
            0xA1: self._handle_flow_loop_end,
            0xA8: self._handle_flow_branch_if,
            0xA9: self._handle_flow_break_loop_if,
            # --- 0xE0-0xFF: Вывод данных ---
            0xE0: self._handle_serialize_object,
            0xF0: self._handle_io_write,
            0xFE: self._handle_exec_return,
            0xFF: self._handle_exec_halt,
        }
        print("--- Интерпретатор готов ---")

    def _get_instruction_length(self, at_ip: int) -> int:
        flag = self.plan[at_ip]
        base = 1  # сам флаг

        if flag in (0x02, 0x04, 0x10, 0x11, 0x12, 0x1F, 0x20, 0x21, 0x22, 0x59, 0x5F,
                    0x60, 0x61, 0x62, 0x68, 0x3B, 0xA0, 0xA8, 0xA9, 0xE0, 0xF0):
            return base + self._instr_lengths.get(flag, 0)

        if flag == 0x2A:  # STRING_FORMAT
            count = self.plan[at_ip + 4]
            return base + 4 + count

        if flag == 0x38:  # TENSOR_MANIPULATE (pad)
            return base + 5   # op(1)+out(1)+in(1)+pad_width(1)+const_val(1)

        if flag == 0x39:  # TENSOR_COMBINE
            count = self.plan[at_ip + 3]
            return base + 3 + count + 1  # + axis

        if flag == 0x3A:  # TENSOR_INFO
            op = self.plan[at_ip + 1]
            return base + 3 + (1 if op == 1 else 0)

        if flag == 0x80:  # MODEL_RUN_STATIC
            count_in = self.plan[at_ip + 2]
            count_out_pos = at_ip + 3 + count_in
            count_out = self.plan[count_out_pos]
            return base + 3 + count_in + 1 + count_out

        if flag == 0xFE:  # EXEC_RETURN
            count = self.plan[at_ip + 1]
            return base + 1 + count

        if flag == 0xA1:  # FLOW_LOOP_END
            return base + 2

        raise NotImplementedError(f"Длина инструкции {hex(flag)} не реализована (ip={at_ip})")

    def _read_u8(self): val = self.plan[self.ip]; self.ip += 1; return val
    def _read_u16(self): val = struct.unpack('<H', self.plan[self.ip:self.ip+2])[0]; self.ip += 2; return val
    def _read_i16(self): val = struct.unpack('<h', self.plan[self.ip:self.ip+2])[0]; self.ip += 2; return val

    def run(self):
        print("\n--- Запуск выполнения MEP плана ---")
        self.running = True
        while self.ip < len(self.plan) and self.running:
            start_ip = self.ip
            flag = self._read_u8()
            handler = self.handlers.get(flag)
            if handler:
                handler()
            else:
                raise NotImplementedError(f"Инструкция с флагом {hex(flag)} по адресу {start_ip} не реализована.")
        print("--- Выполнение MEP плана завершено ---")
        return self.return_value

    def _scalar(self, value: Any) -> Any:
        """Безопасно преобразует значение в Python скаляр, если возможно."""
        if hasattr(value, 'item'):
            return value.item()
        return value

    # --- Обработчики инструкций ---

    def _handle_src_user_prompt(self):
        out_key, _, prompt_const_id = self._read_u8(), self._read_u8(), self._read_u16()
        self.context[out_key] = input(self.constants[prompt_const_id])

    def _handle_src_constant(self):
        out_key, const_id = self._read_u8(), self._read_u16()
        self.context[out_key] = self.constants[const_id]

    def _handle_res_load_model(self):
        model_id, path_const_id = self._read_u8(), self._read_u16()
        model_path = self.constants[path_const_id]
        print(f"MEP: Загрузка модели ID {model_id} из '{model_path}'...")
        self.resources[model_id] = NacRuntime(model_path)

    def _handle_res_load_extern(self):
        """0x12: RES_LOAD_EXTERN — универсальная загрузка внешних компонентов."""
        out_key = self._read_u8()
        res_type = self._read_u8()
        res_id_const_id = self._read_u16()
        res_id_value = self.constants[res_id_const_id]

        if res_type == 0:  # 0 = From NacRuntime / model component
            model_runtime = self.resources.get(res_id_value)
            if model_runtime:
                print(f"MEP: Извлечение внешнего ресурса (NacRuntime) из модели ID {res_id_value}.")
                self.context[out_key] = model_runtime
            else:
                raise RuntimeError(f"RES_LOAD_EXTERN (type=0): модель ID {res_id_value} не найдена.")
        
        elif res_type == 1:  # 1 = Standalone ресурс (по пути/имени)
            print(f"MEP: Загрузка standalone внешнего ресурса типа {res_type} из '{res_id_value}'...")
            # Здесь можно расширять: AutoTokenizer.from_pretrained, diffusers.Scheduler и т.д.
            # Пока просто сохраняем значение (можно заменить на реальную загрузку)
            self.context[out_key] = res_id_value  # или реализовать загрузчик
        
        else:
            raise NotImplementedError(f"RES_LOAD_EXTERN: неизвестный тип {res_type}")
    
    def _handle_res_unload(self):
        res_type, id_or_key = self._read_u8(), self._read_u8()
        if res_type == 0: # model
            if id_or_key in self.resources: del self.resources[id_or_key]
        elif res_type == 1: # context
            self.context[id_or_key] = None

    def _handle_res_load_datafile(self):
        """0x11: Загружает данные из файла в context."""
        out_key, file_type, path_const_id = self._read_u8(), self._read_u8(), self._read_u16()
        file_path = self.constants[path_const_id]
        
        # file_type 2 зарезервирован для .npy
        if file_type == 2:
            try:
                self.context[out_key] = np.load(file_path)
                print(f"MEP: Файл данных '{file_path}' загружен в context[{out_key}].")
            except Exception as e:
                raise IOError(f"Не удалось загрузить .npy файл '{file_path}': {e}")
        else:
            raise NotImplementedError(f"RES_LOAD_DATAFILE для типа файла {file_type} не реализован.")

    def _handle_preproc_encode(self):
        proc_key, in_key, out_key = self._read_u8(), self._read_u8(), self._read_u8()
        self.context[out_key] = self.context[proc_key].encode(self.context[in_key])

    def _handle_preproc_decode(self):
        proc_key, in_key, out_key = self._read_u8(), self._read_u8(), self._read_u8()
        ids = self.context[in_key]
        if isinstance(ids, np.ndarray): ids = ids.flatten().tolist()
        if not isinstance(ids, list): ids = [self._scalar(ids)]
        self.context[out_key] = self.context[proc_key].decode(ids)

    def _handle_preproc_get_id(self):
        proc_key, item_const_id, out_key = self._read_u8(), self._read_u16(), self._read_u8()
        token_str = self.constants[item_const_id]
        token_id = self.context[proc_key].get_token_id(token_str)
        if token_id is None: raise ValueError(f"Токен '{token_str}' не найден.")
        self.context[out_key] = token_id

    def _handle_string_format(self):
        out_key, format_const_id, count = self._read_u8(), self._read_u16(), self._read_u8()
        keys = [self._read_u8() for _ in range(count)]
        self.context[out_key] = self.constants[format_const_id].format(*[self.context[k] for k in keys])

    def _handle_tensor_create(self):
        out_key, dtype_code, creation_type = self._read_u8(), self._read_u8(), self._read_u8()
        np_dtype = {0: np.float32, 5: np.int64}.get(dtype_code, np.float32)
        if creation_type == 0: # from_py
            val = self.context[self._read_u8()]
            self.context[out_key] = np.array(val, dtype=np_dtype).reshape(1, -1)
        elif creation_type == 1: # arange
            self.context[out_key] = np.arange(self.context[self._read_u8()], dtype=np_dtype).reshape(1, -1)
        elif creation_type == 2: # ones
            self.context[out_key] = np.ones(self.context[self._read_u8()], dtype=np_dtype)
        else: raise NotImplementedError(f"TENSOR_CREATE тип {creation_type} не реализован.")

    def _handle_tensor_manipulate(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        tensor = self.context[in_key]
        if op_type == 1: # pad
            pad_width_key, const_val_key = self._read_u8(), self._read_u8()
            pad_width = self._scalar(self.context[pad_width_key])
            const_val = self._scalar(self.context[const_val_key])
            # Left-pad the last dimension
            pad_tuple = [(0, 0)] * (tensor.ndim - 1) + [(pad_width, 0)]
            self.context[out_key] = np.pad(tensor, pad_tuple, 'constant', constant_values=const_val)
        else: raise NotImplementedError(f"TENSOR_MANIPULATE тип {op_type} не реализован.")

    def _handle_tensor_combine(self):
        op_type, out_key, count = self._read_u8(), self._read_u8(), self._read_u8()
        keys = [self._read_u8() for _ in range(count)]
        tensors = [self.context[k] for k in keys]
        if op_type == 0: # concat
            axis_key = self._read_u8()
            axis = self._scalar(self.context[axis_key])
            self.context[out_key] = np.concatenate(tensors, axis=axis)
        else: raise NotImplementedError(f"TENSOR_COMBINE тип {op_type} не реализован.")

    def _handle_tensor_info(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        tensor = self.context[in_key]
        if op_type == 0: self.context[out_key] = tensor.shape
        elif op_type == 1:
            dim_idx = self._scalar(self.context[self._read_u8()])
            self.context[out_key] = tensor.shape[dim_idx]
        elif op_type == 2: self.context[out_key] = tensor.item()
        else: raise NotImplementedError(f"TENSOR_INFO тип {op_type} не реализован.")

    def _handle_tensor_extract(self):
        out_key, in_tensor_key, in_idx_key = self._read_u8(), self._read_u8(), self._read_u8()
        tensor, index = self.context[in_tensor_key], self.context[in_idx_key]
        index = self._scalar(index)
        # Assumes extraction from the last non-batch dimension
        # E.g. logits shape (1, seq_len, vocab_size) -> extract token at `index` from `seq_len`
        if tensor.ndim == 3: extracted = tensor[0, index, :]
        elif tensor.ndim == 2: extracted = tensor[0, index]
        elif tensor.ndim == 1: extracted = tensor[index]
        else: raise TypeError("TENSOR_EXTRACT не поддерживает тензоры с ndim < 1 или > 3.")
        self.context[out_key] = extracted
    
    def _handle_sys_copy(self):
        out_key, in_key = self._read_u8(), self._read_u8()
        self.context[out_key] = self.context[in_key]

    def _handle_sys_debug_print(self):
        key, msg_const_id = self._read_u8(), self._read_u16()
        print(f"[DEBUG] {self.constants[msg_const_id]}: {self.context[key]}")

    def _handle_math_unary(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        if op_type == 0: self.context[out_key] = softmax(self.context[in_key])
        else: raise NotImplementedError(f"MATH_UNARY тип {op_type} не реализован.")

    def _handle_math_binary(self):
        op_type, out_key, key1, key2 = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        val1, val2 = self._scalar(self.context[key1]), self._scalar(self.context[key2])
        ops = [val1 + val2, val1 - val2, val1 * val2]
        self.context[out_key] = ops[op_type]

    def _handle_math_aggregate(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        if op_type == 0: # argmax
            self.context[out_key] = np.argmax(self.context[in_key], axis=-1)
        else: raise NotImplementedError(f"MATH_AGGREGATE тип {op_type} не реализован.")

    def _handle_logic_compare(self):
        op_type, out_key, key1, key2 = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        val1, val2 = self._scalar(self.context[key1]), self._scalar(self.context[key2])
        ops = [val1 == val2, val1 != val2, val1 > val2, val1 < val2]
        self.context[out_key] = ops[op_type]

    def _handle_model_run_static(self):
        model_id = self._read_u8()
        count_in = self._read_u8()
        in_keys = [self._read_u8() for _ in range(count_in)]
        count_out = self._read_u8()
        out_keys = [self._read_u8() for _ in range(count_out)]
        
        inputs = [self.context[k] for k in in_keys]
        outputs = self.resources[model_id].run(inputs)
        
        for i, out_key in enumerate(out_keys): self.context[out_key] = outputs[i]

    def _handle_flow_loop_start(self):
        counter_key = self._read_u8()
        count = self._scalar(self.context[counter_key])
        self.loop_stack.append({'counter_key': counter_key, 'count': count})
        
        if count <= 0:
            # Корректный пропуск тела цикла
            # Ищем соответствующий FLOW_LOOP_END (0xA1), уважая вложенность
            balance = 1
            search_ip = self.ip
            while search_ip < len(self.plan) and balance > 0:
                flag = self.plan[search_ip]
                if flag == 0xA0:
                    balance += 1
                elif flag == 0xA1:
                    balance -= 1
                
                if balance == 0:
                    # Мы нашли парный loop_end, выходим из поиска
                    break
                
                # Перемещаемся на следующую инструкцию, используя ее реальную длину
                search_ip += self._get_instruction_length(search_ip)
            
            # Перемещаем указатель инструкции за найденный FLOW_LOOP_END
            if balance == 0:
                self.ip = search_ip + self._get_instruction_length(search_ip)
            else:
                raise RuntimeError("Не найден парный FLOW_LOOP_END для пропуска цикла.")

    def _handle_flow_loop_end(self):
        jump_offset = self._read_i16()
        if not self.loop_stack: raise RuntimeError("FLOW_LOOP_END без FLOW_LOOP_START.")
        
        loop_info = self.loop_stack[-1]
        loop_info['count'] -= 1
        
        if loop_info['count'] > 0:
            self.ip += jump_offset # Возвращаемся в начало
        else:
            self.loop_stack.pop() # Завершаем цикл

    def _handle_flow_branch_if(self):
        cond_key, jump_offset = self._read_u8(), self._read_i16()
        if self._scalar(self.context[cond_key]):
            self.ip += jump_offset

    def _handle_flow_break_loop_if(self):
        cond_key, jump_offset = self._read_u8(), self._read_i16()
        if not self.loop_stack: raise RuntimeError("FLOW_BREAK_LOOP_IF вне цикла.")
        
        if self._scalar(self.context[cond_key]):
            self.ip += jump_offset
            self.loop_stack.pop() # Выходим из цикла

    def _handle_serialize_object(self):
        """0xE0: Сериализует объект из context."""
        out_key, in_key, format_type = self._read_u8(), self._read_u8(), self._read_u8()
        obj = self.context[in_key]
        
        if format_type == 0: # UTF8_STRING
            self.context[out_key] = str(obj)
        elif format_type == 1: # JSON
            self.context[out_key] = json.dumps(obj)
        elif format_type == 2: # PNG
            if not isinstance(obj, np.ndarray):
                raise TypeError("Для сериализации в PNG ожидается numpy array.")
            img = Image.fromarray(obj.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            self.context[out_key] = buffer.getvalue()
        # ... можно добавить другие форматы (WAV, RAW_TENSOR_BYTES) ...
        else:
            raise NotImplementedError(f"SERIALIZE_OBJECT для типа {format_type} не реализован.")
            
    def _handle_io_write(self):
        """0xF0: Универсальный обработчик вывода."""
        in_key, dest_type, dest_key, write_mode = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        data = self.context[in_key]
        
        if dest_type == 0: # STDOUT
            # stream_mode=2 - это печать без перевода строки
            end_char = "" if write_mode == 2 else "\n"
            print(data, end=end_char, flush=True)
        elif dest_type == 1: # STDERR
            end_char = "" if write_mode == 2 else "\n"
            print(data, file=sys.stderr, end=end_char, flush=True)
        elif dest_type == 2: # FILE
            path = self.context[dest_key]
            # Определяем режим записи файла
            py_mode = 'wb' if isinstance(data, bytes) else 'w'
            if write_mode == 1: # APPEND
                py_mode = 'ab' if isinstance(data, bytes) else 'a'
            
            with open(path, py_mode) as f:
                f.write(data)
        else:
            raise NotImplementedError(f"IO_WRITE для dest_type {dest_type} не реализован.")
            
    def _handle_exec_return(self):
        """0xFE: Завершает выполнение и возвращает значения."""
        count = self._read_u8()
        keys = [self._read_u8() for _ in range(count)]
        if count == 1:
            self.return_value = self.context[keys[0]]
        else:
            self.return_value = [self.context[k] for k in keys]
        self.running = False
        
    def _handle_exec_halt(self):
        """0xFF: Безусловно останавливает выполнение."""
        self.running = False