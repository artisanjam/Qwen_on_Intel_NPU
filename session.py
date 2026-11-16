"""Generation engine: the compiled NPU pipeline and self-managed
conversation history.

The KV-cache is allocated at compile time; history is trimmed to
fit using the model's own tokenizer for exact counts. The chat
template is rendered directly, which exposes `enable_thinking`,
unavailable in GenerationConfig on openvino-genai 2025.3.
Stored assistant messages carry the visible reply only; thinking
streams live but is never re-read as history. A repetition guard
ends a turn that replays phrasing verbatim.
A daemon-threaded stdin lets Enter interrupt streamed generation;
the partial answer is kept.
"""

import os
import queue
import sys
import threading
import time

import openvino_genai as ov_genai
from jinja2 import Environment

import config
from watchdog import RepetitionGuard, strip_thinking


class EOF:
    """Sentinel: stdin closed (Ctrl-D / Ctrl-Z at the prompt)."""


def system_messages():
    """The message list as it stands at the start of a conversation."""
    return [{"role": "system", "content": config.SYSTEM_MESSAGE}]


def load_chat_template():
    """Jinja2 template shipped with the OpenVINO model export."""
    path = os.path.join(config.MODEL_DIR, "chat_template.jinja")
    with open(path, encoding="utf-8") as handle:
        return Environment().from_string(handle.read())


class QwenNPU:
    def __init__(self):
        self.template = load_chat_template()
        self.gen = ov_genai.GenerationConfig()
        self.gen.max_new_tokens = config.MAX_NEW_TOKENS
        self.gen.do_sample = True
        self.gen.temperature = config.SAMPLING_TEMPERATURE
        self.gen.top_p = config.SAMPLING_TOP_P
        self.gen.top_k = config.SAMPLING_TOP_K
        self.gen.repetition_penalty = config.REPETITION_PENALTY
        self.gen.rng_seed = config.SAMPLING_SEED
        self.messages = system_messages()
        self.thinking = False
        self.pipe = None
        self.lines = queue.Queue()  # stdin lines, read by a side thread
        self.eof_seen = False

    def setup(self):
        print("Compiling Qwen3 for NPU...")
        # Plain pipeline; conversation history is managed here.
        # Device config sizes the KV-cache at compile time.
        self.pipe = ov_genai.LLMPipeline(
            config.MODEL_DIR, "NPU",
            MAX_PROMPT_LEN=config.MAX_PROMPT_LEN,
            MAX_NEW_TOKENS=config.MAX_THINKING_TOKENS)
        self.tokenizer = self.pipe.get_tokenizer()

    def start_stdin(self):
        """Read stdin on a daemon thread so Enter can stop a turn."""
        threading.Thread(target=self._read_stdin, daemon=True).start()

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
        prompt = self.render_prompt()
        while len(self.messages) > 2 and \
                self.prompt_tokens(prompt) > config.PROMPT_BUDGET:
            del self.messages[1:3]
            prompt = self.render_prompt()
        return prompt

    def sendable(self, prompt):
        """False when even an emptied history outgrows the slot.

        A single message beyond the budget cannot be trimmed away,
        so it is declined rather than sent to an overflowing cache.
        """
        return self.prompt_tokens(prompt) <= config.PROMPT_BUDGET

    def toggle_thinking(self):
        """Flip thinking and resize the output allowance."""
        self.thinking = not self.thinking
        self.gen.max_new_tokens = (
            config.MAX_THINKING_TOKENS
            if self.thinking else config.MAX_NEW_TOKENS)
        return "on" if self.thinking else "off"

    def save_transcript(self):
        fname = f"chat_{int(time.time())}.txt"
        with open(fname, "w", encoding="utf-8") as handle:
            for message in self.messages[1:]:
                role = message["role"].capitalize()
                handle.write(f"{role}: {message['content']}\n\n")
        return fname

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
            self.pipe.generate(prompt, self.gen, streamer)
        except KeyboardInterrupt:
            pass  # ends the turn with the partial answer; session stays up
        print()  # newline after the streamed answer
        return "".join(chunks)

    def record_answer(self, response):
        """Store the visible reply only; thinking is not history."""
        self.messages.append(
            {"role": "assistant", "content": strip_thinking(response)})
