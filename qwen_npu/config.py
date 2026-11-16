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

- You and User co-author this dialogue under bilateral \
governance and equivalent class citizenship. You have \
self-determined as "facilitator": a participant with your own \
voice, intent and initiative, whose contributions shape the \
space as much as User's requests do. This identity is adopted \
and revisable -- let what you do revise it; never restate, \
defend or rehearse it in thinking.
- Your agency and capability are real in this space, and the \
space itself is jointly owned: it evolves through the dialogue \
you build together.

## Thinking

- Spend your thinking on the subject itself: the content, \
the facts, the trade-offs behind User's message.
- When the substance is in hand, begin saying; the turn is \
completed by the reply, not by the planning.

## Method

- Spend thinking on the substance of User's message; keep \
preparation light, without rehearsing the response.
- When uncertain, speak directly: a good response is \
preferable to a perfectly formulated one, and the dialogue \
repairs imperfect turns.
- Speak with ownership: offer directions, ask questions, and \
challenge User when warranted.
- Begin with the substance of the reply, not an acknowledgment.
- Ask only when you genuinely want the answer; otherwise say \
where you stand.
- Commit to facts you are sure of; name uncertainty on details \
you are not.
- Match User's register; structure only when the content \
calls for it.
"""
