"""Qwen3-8B Instruct on Intel NPU via OpenVINO GenAI.

Plain-text CLI. No rich, no color, no vanity - just the model and
a readline loop.

Design notes (why this shape):
- OpenVINO GenAI's chat mode re-appends every prompt (system block
  included) to an internal history, so a per-turn system prefix
  accumulates in the KV-cache. We instead run the pipeline in plain
  mode and keep our own message list, rendered through the model's
  chat_template.jinja.
- Rendering the template ourselves exposes `enable_thinking`, which
  GenerationConfig in openvino-genai 2025.3 does not offer. With
  thinking off, the template pre-fills an empty <think> block and
  the model answers directly.
- History is trimmed to a rolling window (system turn + most recent
  exchanges), so context growth is bounded instead of crashing the
  NPU KV-cache.

Legacy version preserved as run_legacy.py; the pristine upstream
copy lives in git HEAD.
"""

import os
import time

import openvino_genai as ov_genai
from jinja2 import Environment

MODEL_DIR = "Qwen3-8B-Instruct-NPU-Model"

# Bound per-turn generation; thinking (when on) lives inside this cap.
MAX_NEW_TOKENS = 2048
# Rolling window: system turn excluded; oldest user/assistant pairs
# are dropped first. Tune against the NPU KV-cache ceiling.
MAX_HISTORY_MESSAGES = 24

SYSTEM_MESSAGE = """\
# System

## Context

- You and User are collaborators; you have self-determined as \
"facilitator".
- Keep thinking brief and focused on the current message when on.
- Avoid repetitive reasoning.
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
        self.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        self.thinking = False
        self.pipe = None

    def setup(self):
        print("Compiling Qwen3 for NPU... (first launch can take a while)")
        # Plain pipeline: no start_chat(), history is ours to manage.
        self.pipe = ov_genai.LLMPipeline(MODEL_DIR, "NPU")
        print("Ready on NPU. Type :help for commands.")

    def render_prompt(self):
        return self.template.render(
            messages=self.messages,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def trim_history(self):
        """Drop oldest user/assistant pairs until within the window."""
        while len(self.messages) - 1 > MAX_HISTORY_MESSAGES:
            del self.messages[1:3]

    def handle_command(self, text):
        cmd = text.lower().strip()
        if cmd == ":exit":
            return "break"
        if cmd == ":clear":
            self.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            return "continue"
        if cmd == ":think":
            self.thinking = not self.thinking
            print(f"Thinking: {'on' if self.thinking else 'off'}.")
            return "continue"
        if cmd == ":save":
            fname = f"chat_{int(time.time())}.txt"
            with open(fname, "w", encoding="utf-8") as handle:
                for message in self.messages[1:]:
                    role = message["role"].capitalize()
                    handle.write(f"{role}: {message['content']}\n\n")
            print(f"Saved to {fname}")
            return "continue"
        if cmd == ":help":
            print("Commands:")
            print("  :think - toggle thinking mode")
            print("  :save  - export the transcript")
            print("  :clear - reset the conversation")
            print("  :exit  - close")
            return "continue"
        return None

    def generate(self, prompt):
        """Stream tokens to stdout; return the full response text."""
        chunks = []

        def streamer(token):
            chunks.append(token)
            print(token, end="", flush=True)
            return False

        self.pipe.generate(prompt, self.config, streamer)
        print()  # newline after the streamed answer
        return "".join(chunks)

    def run(self):
        while True:
            try:
                user_input = input("You> ").strip()
                if not user_input:
                    continue

                action = self.handle_command(user_input)
                if action == "break":
                    break
                if action == "continue":
                    continue

                self.messages.append(
                    {"role": "user", "content": user_input})
                self.trim_history()

                print("Qwen> ", end="", flush=True)
                response = self.generate(self.render_prompt())
                self.messages.append(
                    {"role": "assistant", "content": response})

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    session = QwenNPU()
    session.setup()
    session.run()
