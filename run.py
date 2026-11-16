"""Qwen3-8B Instruct on Intel NPU via OpenVINO GenAI.

The pipeline runs in plain mode with self-managed conversation
history, rendered through the model's chat_template.jinja. The NPU
KV-cache is allocated at compile time from MAX_PROMPT_LEN and
MAX_NEW_TOKENS; history is trimmed to fit using the model's own
tokenizer for exact counts, so neither overflow.
Rendering the template directly exposes `enable_thinking`, which
GenerationConfig does not offer in openvino-genai 2025.3. When
thinking is off, the template pre-fills an empty <think> block and
the model answers directly.
A repetition watchdog ends a turn that emits a sentence it already
said this turn, or replays the previous answer verbatim -- the
signature of a copy-attractor loop.
A daemon thread owns stdin, so Enter interrupts streamed generation
and the Ctrl key that closes stdin (Ctrl-D, Ctrl-Z on Windows)
exits the session.
"""

import os
import queue
import re
import sys
import threading
import time

import openvino_genai as ov_genai
from jinja2 import Environment

MODEL_DIR = "Qwen3-8B-Instruct-NPU-Model"

# Cache slots fixed at NPU compile time. Prompts beyond
# MAX_PROMPT_LEN are rejected; truncated generation appears when
# the cache cannot hold prompt plus output.
MAX_PROMPT_LEN = 4096
MAX_NEW_TOKENS = 2048
MAX_THINKING_TOKENS = 8192
PROMPT_RESERVE = 512  # headroom left free of history in the slot

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

SENTENCE_END = re.compile(r"[.!?]")


class EOF:
    """Sentinel: stdin closed (Ctrl-D / Ctrl-Z at the prompt)."""


def normalize(part):
    """Lowercase, whitespace collapsed: casing is not a loop signal."""
    return " ".join(part.lower().split())


def sentences(text):
    """Normalized sentences long enough to be loop material."""
    return [s for s in map(normalize, SENTENCE_END.split(text or ""))
            if len(s) >= WATCH_MIN_CHARS]


class RepetitionGuard:
    """End a turn that repeats itself verbatim.

    Within-turn: a tracked sentence emitted a second time ends
    the turn. Cross-turn: tracked sentences replayed from the
    previous answer are counted -- quoting is legitimate once, so
    WATCH_PRIOR_MATCHES distinct replays are needed to call it a
    loop.
    """

    def __init__(self, prior):
        self.prior = set(sentences(prior))
        self.seen = set()
        self.prior_hits = 0
        self.buffer = ""

    def observe(self, token):
        """True once the turn should stop."""
        self.buffer += token
        parts = SENTENCE_END.split(self.buffer)
        self.buffer = parts[-1]  # keep the unterminated tail
        for part in parts[:-1]:
            sentence = normalize(part)
            if len(sentence) < WATCH_MIN_CHARS:
                continue
            if sentence in self.seen:
                return True
            if sentence in self.prior:
                self.prior_hits += 1
                if self.prior_hits >= WATCH_PRIOR_MATCHES:
                    return True
            self.seen.add(sentence)
        return False


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


def load_chat_template():
    """Jinja2 template shipped with the OpenVINO model export."""
    path = os.path.join(MODEL_DIR, "chat_template.jinja")
    with open(path, encoding="utf-8") as handle:
        return Environment().from_string(handle.read())


class QwenNPU:
    def __init__(self):
        self.template = load_chat_template()
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = MAX_NEW_TOKENS
        self.config.do_sample = True
        self.config.temperature = SAMPLING_TEMPERATURE
        self.config.top_p = SAMPLING_TOP_P
        self.config.top_k = SAMPLING_TOP_K
        self.config.repetition_penalty = REPETITION_PENALTY
        self.config.rng_seed = SAMPLING_SEED
        self.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        self.thinking = False
        self.pipe = None
        self.lines = queue.Queue()  # stdin lines, read by a side thread
        self.eof_seen = False

    def setup(self):
        print("Compiling Qwen3 for NPU...")
        # Plain pipeline; conversation history is managed here.
        # Device config sizes the KV-cache at compile time.
        self.pipe = ov_genai.LLMPipeline(
            MODEL_DIR, "NPU",
            MAX_PROMPT_LEN=MAX_PROMPT_LEN,
            MAX_NEW_TOKENS=MAX_THINKING_TOKENS)
        self.tokenizer = self.pipe.get_tokenizer()
        print("Ready on NPU. Type :help for commands.")

    def render_prompt(self):
        return self.template.render(
            messages=self.messages,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def prompt_tokens(self, prompt):
        """Exact token count from the model's tokenizer."""
        return self.tokenizer.encode(prompt).input_ids.shape[-1]

    def trim_to_prompt_budget(self):
        """Drop oldest pairs until the prompt fits the cache slot."""
        budget = MAX_PROMPT_LEN - PROMPT_RESERVE
        prompt = self.render_prompt()
        while len(self.messages) > 2 and self.prompt_tokens(prompt) > budget:
            del self.messages[1:3]
            prompt = self.render_prompt()
        return prompt

    def handle_command(self, text):
        """Map a colon command to the outcome it requests.

        "exit" ends the session, "command" means the command was
        handled, None means the text is an ordinary message.
        """
        cmd = text.lower().strip()
        if cmd == ":exit":
            return "exit"
        if cmd == ":clear":
            self.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            return "command"
        if cmd == ":think":
            self.thinking = not self.thinking
            self.config.max_new_tokens = (
                MAX_THINKING_TOKENS if self.thinking else MAX_NEW_TOKENS)
            state = "on" if self.thinking else "off"
            print(f"Thinking: {state}.")
            return "command"
        if cmd == ":save":
            fname = self.save_transcript()
            print(f"Saved to {fname}")
            return "command"
        if cmd == ":help":
            self.print_help()
            return "command"
        return None

    def save_transcript(self):
        fname = f"chat_{int(time.time())}.txt"
        with open(fname, "w", encoding="utf-8") as handle:
            for message in self.messages[1:]:
                role = message["role"].capitalize()
                handle.write(f"{role}: {message['content']}\n\n")
        return fname

    def print_help(self):
        print("Commands:")
        print("  :think - toggle thinking mode")
        print("  :save  - export the transcript")
        print("  :clear - reset the conversation")
        print("  :exit  - close")
        print("Enter or Ctrl-C stops an answer mid-stream; the partial")
        print("answer is kept and the session continues. Ctrl-C at the")
        print("prompt exits.")

    def _read_stdin(self):
        """Own stdin so typed input can interrupt streamed generation."""
        while True:
            line = sys.stdin.readline()
            if line == "":  # Ctrl-D (POSIX), Ctrl-Z at line start (Windows)
                self.lines.put(EOF)
                return
            self.lines.put(line)

    def interrupted(self):
        """True once Enter was pressed while tokens were streaming."""
        signaled = False
        try:
            while True:  # drain so stray Enters do not leak to the prompt
                line = self.lines.get_nowait()
                signaled = True
                if line is EOF:
                    self.eof_seen = True
        except queue.Empty:
            pass
        return signaled

    def previous_answer(self):
        """Most recent assistant message, for the cross-turn guard."""
        for message in reversed(self.messages):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    def generate(self, prompt):
        """Stream tokens to stdout; return the response so far.

        The streamer polls the stdin queue each token, so Enter (or
        Ctrl-D followed by Enter on Windows) stops generation
        without ending the session; the partial answer is kept.
        Ctrl-C also ends the turn here, not the session. The
        repetition guard ends the turn on verbatim self-replay,
        with a visible notice so the gate can count events.
        """
        chunks = []
        guard = RepetitionGuard(self.previous_answer())

        def streamer(token):
            chunks.append(token)
            print(token, end="", flush=True)
            if self.interrupted():
                return True
            if guard.observe(token):
                print("\n[stopped: repeated phrasing]", flush=True)
                return True
            return False

        try:
            self.pipe.generate(prompt, self.config, streamer)
        except KeyboardInterrupt:
            pass  # ends the turn with the partial answer; session stays up
        print()  # newline after the streamed answer
        return "".join(chunks)

    def run(self):
        threading.Thread(target=self._read_stdin, daemon=True).start()
        while True:
            try:
                print("\n---\nYou> ", end="", flush=True)
                line = self.lines.get()
                if line is EOF:
                    break
                user_input = line.strip()
                if not user_input:
                    continue

                action = self.handle_command(user_input)
                if action == "exit":
                    break
                if action == "command":
                    continue

                self.messages.append(
                    {"role": "user", "content": user_input})
                prompt = self.trim_to_prompt_budget()

                print("\n---\nAgent> ", end="", flush=True)
                response = self.generate(prompt)
                self.messages.append(
                    {"role": "assistant", "content": response})
                if self.eof_seen:
                    break

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    session = QwenNPU()
    session.setup()
    session.run()