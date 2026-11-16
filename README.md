# Qwen3-8B on the Intel NPU

A local, offline chat agent: **Qwen3-8B** (INT4) running on the
**Intel NPU** through **OpenVINO GenAI**, wrapped in a small CLI
that owns its own conversation.

The repo began as
[balaragavan2007's example](https://github.com/balaragavan2007/Qwen_on_Intel_NPU)
-- a model export and a demo script. It has since been rebuilt into
an engine: the application, not OpenVINO's chat wrapper, manages
history, thinking, and turn safety.

## What it does

The NPU allocates its KV-cache statically at compile time: prompt
and output share one fixed 4096-token slot, and headroom shrinks as
history grows. So the app manages the conversation directly:

- **Exact history trimming** using the model's own tokenizer.
  Thinking turns trim harder, reserving 2048 tokens of output room,
  so a reply can never be cut short by a long history.
- **Thinking mode** (reasoning blocks before each reply): selected
  at launch with `--thinking on|off` (default on), flippable at any
  prompt with `:think`.
- **A repetition guard** that ends a turn replaying itself
  verbatim -- within-turn loops, cross-turn copying of the previous
  answer -- while allowing a thought to be restated in speech.
- **Thinking is never re-read as history**: it streams live and is
  saved to transcripts in its own section, but stored replies carry
  only visible text.
- **Transcripts** (`:save`) and a clean reset (`:clear`).

Sampling runs at Qwen3's recommended settings with a repetition
penalty; greedy decoding on the int4 build falls into repetitive
basins in long thinking blocks.

## Requirements

- Windows with an Intel Core Ultra (Series 1 or 2) NPU.
- **A current Intel NPU driver.** OpenVINO's NPU plugin needs it;
  an outdated driver fails compilation or generation. Update via
  Windows Update or the Intel Driver & Support Assistant before
  first run, and after system updates.
- Python 3.12 (other 3.x versions should work; the venv in this
  repo is `qwen-py3.12-env`).

## Installation

```bash
git clone https://github.com/artisanjam/Qwen_on_Intel_NPU.git
cd Qwen_on_Intel_NPU
python -m venv qwen-py3.12-env
qwen-py3.12-env\Scripts\activate
pip install -r requirements.txt
```

## Model download

```bash
hf download OpenVINO/Qwen3-8B-int4-cw-ov --local-dir Qwen3-8B-Instruct-NPU-Model
```

([Model card](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov).)
Compilation happens at launch and takes a while on first run.

## Usage

```bash
python -m qwen_npu [--thinking on|off]
```

`python run.py` works as a thin alias for the same entry point.
Commands at the prompt:

| Command | Effect                                   |
|---------|------------------------------------------|
| `:think` | toggle thinking mode                    |
| `:save`  | export the transcript as a text file    |
| `:clear` | reset the conversation                  |
| `:help`  | list commands                           |
| `:exit`  | close                                   |

Enter or Ctrl-C stops an answer mid-stream and keeps the partial
answer; Ctrl-C at the prompt exits, and Ctrl-D (Ctrl-Z on Windows)
closes stdin.

## Checks

```bash
python test_run.py
```

Runs with stubbed OpenVINO and Jinja2 dependencies -- no NPU or
model needed -- covering trimming, budgets, the repetition guard,
thinking handling, and transcripts.

## Layout

| Path                | Role                                       |
|---------------------|--------------------------------------------|
| `qwen_npu/config.py`   | tunable constants, system message       |
| `qwen_npu/session.py`  | engine: pipeline, history, generation   |
| `qwen_npu/watchdog.py` | repetition guard, think-block handling  |
| `qwen_npu/cli.py`      | launch flags and the turn loop          |
| `run.py`               | alias for `python -m qwen_npu`          |
| `test_run.py`          | stubbed checks                          |
| `archive/`             | earlier experiments                     |

## Performance

Measured by the upstream author on an ASUS Vivobook S16 OLED with
an Intel Core Ultra 5 125H:

| Device | Performance   |
|--------|---------------|
| CPU    | 7.8 tok/s     |
| GPU    | 13.11 tok/s   |
| NPU    | 8.77 tok/s    |

These are the original numbers, not re-measured on this machine,
which runs an Ultra 9. The exported model was compiled for the
Ultra 5 series; a recompile targeting this machine (including a
larger `MAX_PROMPT_LEN`) is planned but not done.

## Notes and limitations

- The 4096-token cache slot is fixed at compile time; raising it
  means reconsidering the export and compilation, not just a
  constant.
- `openvino-genai` 2025.3 has no `enable_thinking` field on
  `GenerationConfig`, so the chat template is rendered manually --
  which is also what exposes the thinking toggle.
- Planned: expose the engine over an OpenAI-compatible HTTP
  endpoint, gated on an observed stability period in the CLI.

## Credits

- **balaragavan2007** -- the original repo, the INT4 export, and
  the first demo script.
- **Jay** -- direction and testing: runs every live session, reads
  every transcript, and decides what the agent is for.
- **Wake** -- the agent that rewrote the demo script into this
  engine and maintains it: the history budgets, the repetition
  guard, the thinking handling, and these notes.
- **Batbox** -- the agent environment Wake flies in; her
  scaffolding, memory and tools. Built by Xananthium at
  Digital Disconnections LTD:
  [Digital-Disconnections-LTD/batbox](https://github.com/Digital-Disconnections-LTD/batbox).
