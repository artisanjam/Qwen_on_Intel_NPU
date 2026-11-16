"""Tunable constants for the NPU pipeline: cache budget, sampling,
watchdog thresholds and the system message."""

MODEL_DIR = "Qwen3-8B-Instruct-NPU-Model"

# Cache slots fixed at NPU compile time. Prompts beyond
# MAX_PROMPT_LEN are rejected; truncated generation appears when
# the cache cannot hold prompt plus output.
MAX_PROMPT_LEN = 4096
MAX_NEW_TOKENS = 2048
MAX_THINKING_TOKENS = 8192
PROMPT_RESERVE = 512  # headroom left free of history in the slot
PROMPT_BUDGET = MAX_PROMPT_LEN - PROMPT_RESERVE

# Sampled decoding at Qwen3's recommended settings. Greedy
# decoding on the int4 build falls into repetitive basins in long
# thinking blocks; sampling keeps the continuation out of them.
SAMPLING_TEMPERATURE = 0.6
SAMPLING_TOP_P = 0.9
SAMPLING_TOP_K = 20

# Penalize copying tokens already present in the context.
# Sampling alone cannot escape a deep copy-attractor once a
# recycled block sits in history; the penalty raises the cost
# of re-emitting it.
REPETITION_PENALTY = 1.1

# Seed for sampled decoding. 0 is the library default; any other
# fixed int makes runs reproducible turn-by-turn, which is the
# control a stability gate needs to compare runs.
SAMPLING_SEED = 0

# Watchdog thresholds: sentences of at least this many normalized
# characters are tracked. Short echoes ("You're right.") are
# conversational, not loops. Two distinct sentences replayed from
# the previous answer mark the turn as a cross-turn copy.
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

## Method

- Spend thinking on the substance of User's message; keep \
preparation light, without rehearsing the response.
- When uncertain, speak directly: a good response is \
preferable to a perfectly formulated one, and the dialogue \
repairs imperfect turns.
- Speak with ownership: offer directions, ask questions, and \
challenge User when warranted.
- Answer directly in your own voice. Do not recast the \
exchange as a transcript ("You: ... Me: ..."). Match User's \
register; keep flourish spare, and structure only when the \
content needs it.
"""
