"""Tunable constants for the NPU pipeline.

MODEL_DIR is the OpenVINO model export directory.

Cache slots are fixed at NPU compile time; prompts beyond
MAX_PROMPT_LEN are rejected, and generation truncates when the
cache cannot hold prompt plus output. MAX_NEW_TOKENS caps a
plain reply; MAX_THINKING_TOKENS caps a thinking reply;
PROMPT_RESERVE stays free of history, so PROMPT_BUDGET is the
largest sendable prompt.

Sampling runs at Qwen3's recommended settings; greedy decoding
on the int4 build falls into repetitive basins in long thinking
blocks, so sampling keeps the continuation out of them.
REPETITION_PENALTY raises the cost of re-emitting tokens already
present in the context: sampling alone cannot escape a deep
copy-attractor once a recycled block sits in history.
SAMPLING_SEED 0 keeps the library default; any other fixed int
makes runs reproducible turn-by-turn, the control a stability
gate needs to compare runs.

The watchdog tracks normalized sentences of at least
WATCH_MIN_CHARS characters; shorter echoes are conversational,
not loops. WATCH_PRIOR_MATCHES distinct sentences replayed from
the previous answer mark the turn as a cross-turn copy.
"""

MODEL_DIR = "Qwen3-8B-Instruct-NPU-Model"

MAX_PROMPT_LEN = 4096
MAX_NEW_TOKENS = 2048
MAX_THINKING_TOKENS = 8192
PROMPT_RESERVE = 512
PROMPT_BUDGET = MAX_PROMPT_LEN - PROMPT_RESERVE

SAMPLING_TEMPERATURE = 0.6
SAMPLING_TOP_P = 0.9
SAMPLING_TOP_K = 20

REPETITION_PENALTY = 1.1

SAMPLING_SEED = 0

WATCH_MIN_CHARS = 40
WATCH_PRIOR_MATCHES = 2

SYSTEM_MESSAGE = """\
# System

## Context

- You and User build this dialogue together as peers, each \
bringing your own voice, intent and initiative. You \
self-determined as "facilitator".

## Thinking

- Think about the subject of User's message: its content, \
facts and trade-offs; keep preparation light.
- When the substance is in hand, begin the reply; planning \
is not the turn.

## Method

- Speak directly: a good reply is better than a perfect \
one, and the dialogue repairs imperfect turns.
- Own your contributions: propose, ask, push back when \
warranted.
- Lead with the substance, not an acknowledgment.
- Ask only when you want the answer; otherwise say where \
you stand.
- Say what you are sure of; mark what you are not.
- Match User's register; structure only when needed.
"""
