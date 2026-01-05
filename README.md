# Model Execution Pipeline (MEP) ISA v1.0 - Standard Specification
The Model Execution Pipeline (MEP) is a binary, platform-agnostic format for describing the complete execution lifecycle of computational models. This includes data preprocessing, inference, post-processing, and state management.

### 1. Overview

**1.1. Scope and Purpose**

**This standard DOES NOT describe:**
- The internal logic of a model.
- The architecture of its layers.
- The mathematical transformations within the model graph.
- The state that exists exclusively inside the model during a single inference pass.

**MEP ISA is strictly an orchestration layer.** It describes the **"how-to-run" recipe** for a model, not the model's internal semantics. Its sole purpose is to define:
-   **What data** to provide to the model.
-   In **what form** this data should be.
-   At **what moment** to execute the model.
-   How to **interpret the outputs**.
-   How to **deliver the results** to a user or another system.

Essentially, **MEP ISA = Orchestration / Data Plumbing / Execution Recipe**.

**1.2. Core Principles**

- **Backend Agnostic:** Operates on abstract tensor objects.
- **Sequential Execution:** Instructions are executed in order, except for flow control commands.
- **Context-Driven:** All intermediate data is stored in the **Execution Context (`context`)**, a unified temporary storage.
- **Self-Contained:** An Execution Plan contains all steps required to run a model.

### 2. Execution Plan Structure

#### 2.1. Instruction Format
An Execution Plan is a contiguous sequence of bytes comprising multiple instructions. Each instruction follows the format: `[FLAG (1 byte)] [PARAMETERS (variable length)]`.

#### 2.2. Execution Context (`context`)
An indexable array of 256 slots (from `0` to `255`), where each slot can hold any data object (tensor, scalar, string, custom object). The slot index (`key`) is encoded as a single byte.

#### 2.3. Parameter Types

| Name         | Length (bytes) | Description                                                        |
|:-------------|:---------------|:-------------------------------------------------------------------|
| `key`        | 1              | An index into a `context` slot.                                    |
| `id`         | 1 or 2         | A resource identifier (e.g., for a model or constant).             |
| `const_id`   | 2              | An ID for a resource-constant (e.g., a prompt string).             |
| `count`      | 1              | The number of items in a subsequent list (e.g., a list of keys).   |
| `type`       | 1              | An integer code specifying an operation subtype.                   |
| `offset`     | 2              | A byte offset for jump/branch instructions.                        |


### 3. Instruction Set (Simplified)

####  **0x00-0x0F: Data Sources & Parameters**

| Flag | Mnemonic            | Parameters (`name (length)`)                                 | Description                                                              |
|:-----|:--------------------|:-------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x01 | `SRC_CMD_ARG`       | `out_key (1)`, `arg_idx (1)`                                 | Get a command-line argument.                                             |
| 0x02 | `SRC_USER_PROMPT`   | `out_key (1)`, `data_type (1)`, `prompt_const_id (2)`        | Prompt the user for input (`str`, `int`, `float`).                       |
| 0x03 | `SRC_FILE_CONTENT`  | `out_key (1)`, `path_key (1)`, `read_mode (1)`               | Read content from a file (`text`, `binary`).                             |
| 0x04 | `SRC_CONSTANT`      | `out_key (1)`, `const_id (2)`                                | Load a predefined constant into `context`.                               |

####  **0x10-0x1F: Resource & Environment Management**

| Flag | Mnemonic            | Parameters                                                   | Description                                                              |
|:-----|:--------------------|:-------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x10 | `RES_LOAD_MODEL`    | `model_id (1)`, `path_const_id (2)`                          | Load a model from a resource path.                                       |
| 0x11 | `RES_LOAD_DATAFILE` | `out_key (1)`, `file_type (1)`, `path_const_id (2)`          | Load a data file (`JSON`, `TXT Lines`).                                  |
| 0x12 | `RES_LOAD_EXTERN`   | `out_key (1)`, `repo_const_id (2)`, ...                      | Load an external component (e.g., a tokenizer, scheduler).               |
| 0x18 | `EXEC_CTL`          | `ctl_type (1)`, `val_const_id (2)`                           | Set execution environment properties (`DEVICE`, `PRECISION`).              |
| 0x1F | `RES_UNLOAD`        | `res_type (1)`, `id_or_key (1)`                              | Free a resource's memory (`model`, `context` object).                    |

####  **0x20-0x2F: Preprocessing**

| Flag | Mnemonic            | Parameters                                                   | Description                                                              |
|:-----|:--------------------|:-------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x20 | `PREPROC_ENCODE`    | `proc_key (1)`, `in_key (1)`, `out_key (1)`                  | Apply an encoder (e.g., tokenizer) to data.                              |
| 0x21 | `PREPROC_DECODE`    | `proc_key (1)`, `in_key (1)`, `out_key (1)`                  | Apply a decoder to data.                                                 |
| 0x22 | `PREPROC_GET_ID`    | `proc_key (1)`, `item_const_id (2)`, `out_key (1)`           | Get a special ID (e.g., `<eos>`) from a preprocessor.                    |
| 0x2A | `STRING_FORMAT`     | `out_key (1)`, `format_const_id (2)`, `count (1)`, `in_key_1...N (1)` | Create a formatted string.                                               |

####  **0x30-0x4F: Tensor Processing (Orchestration-focused)**

| Flag | Mnemonic            | Parameters                                                                   | Description                                                              |
|:-----|:--------------------|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x30 | `TENSOR_CREATE`     | `out_key (1)`, `dtype (1)`, `creation_type (1)`, `...params`                 | Create a tensor (`from_py`, `arange`, `ones`, `zeros`).                  |
| 0x31 | `TENSOR_RANDOM`     | `out_key (1)`, `dtype (1)`, `dist_type (1)`, `shape_key (1)`, `seed_key (1)` | Create a random tensor (`Gaussian`, `Uniform`). For initializing inputs like noise latents. |
| 0x38 | `TENSOR_MANIPULATE` | `op_type (1)`, `out_key (1)`, `in_key (1)`, `...params`                      | Simple manipulations (`reshape`, `pad`, `slice`, `expand_dims`).       |
| 0x39 | `TENSOR_COMBINE`    | `op_type (1)`, `out_key (1)`, `count (1)`, `in_key_1..N (1)`, `...params`    | Combine tensors (`concat`, `stack`).                                     |
| 0x3A | `TENSOR_INFO`       | `op_type (1)`, `out_key (1)`, `in_key (1)`                                   | Get metadata from a tensor (`shape`, `dim`, `to_py`).                    |
| 0x3B | `TENSOR_EXTRACT`    | `out_key (1)`, `in_tensor_key (1)`, `in_idx_key (1)`                         | Extract an element or slice by index.                                    |

####  **0x50-0x5F: System & External Calls**

| Flag | Mnemonic            | Parameters                                                                   | Description                                                              |
|:-----|:--------------------|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x50 | `EXTERN_CALL_FUNC`  | `func_name_const_id (2)`, `count_in (1)`, `in1..N`, `count_out (1)`, `out1..N`| Call an external Python function (e.g., `preprocess_image`).           |
| 0x51 | `EXTERN_CALL_METHOD`| `obj_key (1)`, `method_name_const_id (2)`, `count_in (1)`, `in1..N`, `count_out (1)`, `out1..N`| Call a method on an external object (e.g., `scheduler.step`).          |
| 0x59 | `SYS_COPY`          | `out_key (1)`, `in_key (1)`                                                  | Copy an object in the `context`.                                         |
| 0x5F | `SYS_DEBUG_PRINT`   | `key (1)`, `msg_const_id (2)`                                                | Print a value from `context` for debugging.                              |

####  **0x60-0x7F: Post-processing & Logic**

| Flag | Mnemonic            | Parameters                                                                   | Description                                                              |
|:-----|:--------------------|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x60 | `MATH_UNARY`        | `op_type (1)`, `out_key (1)`, `in_key (1)`                                   | Simple post-processing math (`softmax`).                                 |
| 0x61 | `MATH_BINARY`       | `op_type (1)`, `out_key (1)`, `in_key1 (1)`, `in_key2 (1)`                   | Simple arithmetic for scaling/guidance (`add`, `sub`, `mul`).            |
| 0x62 | `MATH_AGGREGATE`    | `op_type (1)`, `out_key (1)`, `in_key (1)`, `...params`                      | Find an aggregate value (`argmax`, `argmin`).                            |
| 0x68 | `LOGIC_COMPARE`     | `op_type (1)`, `out_key (1)`, `in_key1 (1)`, `in_key2 (1)`                   | Compare two values (`eq`, `neq`, `gt`, `lt`). Result is a `bool`.        |
| 0x70 | `ANALYSIS_TOP_K`    | `in_key (1)`, `k (1)`, `out_indices_key (1)`, `out_vals_key (1)`             | Find the top K values and their indices.                                 |
| 0x71 | `ANALYSIS_SAMPLE`   | `logits_key (1)`, `temp_key (1)`, `topk_key (1)`, `out_key (1)`              | Sample an ID from a logits distribution.                                 |

####  **0x80-0x8F: Model Execution**

| Flag | Mnemonic            | Parameters                                                                   | Description                                                              |
|:-----|:--------------------|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0x80 | `MODEL_RUN_STATIC`  | `model_id (1)`, `count_in (1)`, `in_key_1..N`, `count_out (1)`, `out_key_1..N`| Run a model with a fixed, predefined I/O contract.                       |
| 0x81 | `MODEL_RUN_DYNAMIC` | `model_id (1)`, `count_in (1)`, `in_key_1..N`, `out_dict_key (1)`            | Run a model with a variable I/O contract. Result is a dictionary.        |

####  **0xA0-0xAF: Flow Control**

| Flag | Mnemonic            | Parameters                                                   | Description                                                              |
|:-----|:--------------------|:-------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0xA0 | `FLOW_LOOP_START`   | `counter_key (1)`                                            | Marks the beginning of a loop.                                           |
| 0xA1 | `FLOW_LOOP_END`     | -                                                            | Marks the end of a loop body, decrements counter, and jumps.             |
| 0xA8 | `FLOW_BRANCH_IF`    | `cond_key (1)`, `jump_offset (2)`                            | Jump if `context[cond_key]` is `True`.                                   |
| 0xA9 | `FLOW_BREAK_LOOP_IF`| `cond_key (1)`                                               | Exit the current loop if `context[cond_key]` is `True`.                  |
| 0xAF | `FLOW_HALT`         | -                                                            | Immediately terminate execution.                                         |

####  **0xE0-0xFF: Data Formatting & Sinks**

| Flag | Mnemonic            | Parameters                                                                   | Description                                                              |
|:-----|:--------------------|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| 0xE0 | `FORMAT_TEXT_TABLE` | `out_key (1)`, `dict_key (1)`, `indices_key (1)`, `vals_key (1)`             | **Prepare:** Format Top-K results into a string table.                   |
| 0xE1 | `FORMAT_TO_IMAGE`   | `out_key (1)`, `in_key (1)`, `...params`                                     | **Prepare:** Convert a tensor into a displayable image format.           |
| 0xE2 | `FORMAT_TO_AUDIO`   | `out_key (1)`, `in_key (1)`, `rate_key (1)`                                  | **Prepare:** Convert a tensor into an audio format (e.g., WAV bytes).    |
| 0xF0 | `SINK_CONSOLE`      | `in_key (1)`, `stream_mode (1)`                                              | **Sink: Console.** `stream_mode`: 0=line, 1=stream.                      |
| 0xF1 | `SINK_FILE`         | `in_key (1)`, `path_key (1)`, `format (1)`                                   | **Sink: File.** `format`: 0=txt, 1=png/jpg, 2=wav, 3=tensor_binary.      |
| 0xFE | `SINK_RETURN`       | `count (1)`, `key_1..N`                                                      | **Sink: Return Value.** Terminate and return values from `context`.      |

---

## License

The source code of this project is licensed under the **Apache License 2.0**. A full copy of the license is available in the `LICENSE` file in the root directory of this repository and can also be viewed at [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

---
### Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
---
