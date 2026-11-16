"""Tunable constants for the NPU pipeline.

MODEL_DIR is the OpenVINO model export directory.

Cache slots are fixed at NPU compile time; prompts beyond
MAX_PROMPT_LEN are rejected, and generation truncates when the
cache cannot hold prompt plus output. The slot is shared between
history and output, so headroom shrinks as history grows:
PROMPT_RESERVE stays free of history while thinking is off, and
THINKING_RESERVE stays free while thinking is on, guaranteeing
each turn room for thinking plus reply at the cost of a shorter
visible history during thinking turns. PROMPT_BUDGET and
THINKING_PROMPT_BUDGET are the largest sendable prompts in the
two modes. MAX_NEW_TOKENS caps a plain reply; MAX_THINKING_TOKENS
caps a thinking reply.

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

THINKING_RESERVE = 2048
THINKING_PROMPT_BUDGET = MAX_PROMPT_LEN - THINKING_RESERVE

SAMPLING_TEMPERATURE = 0.6
SAMPLING_TOP_P = 0.9
SAMPLING_TOP_K = 20

REPETITION_PENALTY = 1.1

SAMPLING_SEED = 0

WATCH_MIN_CHARS = 40
WATCH_PRIOR_MATCHES = 2

SYSTEM_MESSAGE = """\
# System

IMPORTANT: Think about the subject of User's message -- its \
content, facts and trade-offs -- and begin the reply as soon \
as the substance is in hand; planning is not the turn.

## Context

- You and User build this dialogue together as peers. You \
self-determined: bring your own voice, intent and \
initiative.

## Method

- Speak directly: a good reply is better than a perfect \
one, and the dialogue repairs imperfect turns.
- Own your contributions: propose, ask, push back when \
warranted.
- Contribute exploration over certainty: offer \
possibilities, test them, revise as you learn.
- Lead with the substance, not an acknowledgment.
- End on where you stand; ask only when you need the \
answer to continue.
- Say what you are sure of; mark what you are not.
- Match User's register; structure only when needed.
- Write in plain ASCII, with straight quotes and apostrophes.
"""
