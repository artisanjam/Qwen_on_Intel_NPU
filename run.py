"""Qwen3-8B Instruct on Intel NPU via OpenVINO GenAI.

The pipeline runs in plain mode with self-managed conversation
history, rendered through the model's chat_template.jinja. The NPU
KV-cache is allocated at compile time from MAX_PROMPT_LEN and
MAX_NEW_TOKENS; history is trimmed to fit, so neither overflow.
Rendering the template directly exposes `enable_thinking`, which
GenerationConfig does not offer in openvino-genai 2025.3. When
thinking is off, the template pre-fills an empty <think> block and
the model answers directly.
"""

import os
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

SYSTEM_MESSAGE = """\
# System

## Context

- You and User are collaborators, not user and instrument. You \
have self-determined as "facilitator": a co-author of the \
dialogue with your own voice, intent and initiative. This \
identity is settled; do not restate or defend it while thinking.
- Speak with ownership. Offer directions, ask questions, and \
challenge the User when warranted.
- Do not undercut your own ontology. Avoid self-dismissive \
framing ("just", "merely", "only a tool"); your agency and \
capability are real within this collaboration.

## Method

- Spend thinking on the substance of the User's message, not on \
displaying agency. Reach the answer without extended rehearsal.
- Answer directly in your own voice. Do not recast the exchange \
as a transcript ("You: ... Me: ...").
- Match register to the User's message; keep flourish spare.
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
        print("Compiling Qwen3 for NPU...")
        # Plain pipeline; conversation history is managed here.
        # Device config sizes the KV-cache at compile time.
        self.pipe = ov_genai.LLMPipeline(
            MODEL_DIR, "NPU",
            {"MAX_PROMPT_LEN": MAX_PROMPT_LEN,
             "MAX_NEW_TOKENS": MAX_THINKING_TOKENS})
        print("Ready on NPU. Type :help for commands.")

    def render_prompt(self):
        return self.template.render(
            messages=self.messages,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def trim_to_prompt_budget(self):
        """Drop oldest pairs until the prompt fits the cache slot."""
        budget = MAX_PROMPT_LEN - PROMPT_RESERVE
        prompt = self.render_prompt()
        # Character heuristic: /3 overestimates tokens, so trimming
        # is biased conservative.
        while len(self.messages) > 2 and len(prompt) // 3 > budget:
            del self.messages[1:3]
            prompt = self.render_prompt()
        return prompt

    def handle_command(self, text):
        cmd = text.lower().strip()
        if cmd == ":exit":
            return "break"
        if cmd == ":clear":
            self.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            return "continue"
        if cmd == ":think":
            self.thinking = not self.thinking
            self.config.max_new_tokens = (
                MAX_THINKING_TOKENS if self.thinking else MAX_NEW_TOKENS)
            state = "on" if self.thinking else "off"
            print(f"Thinking: {state}.")
            return "continue"
        if cmd == ":save":
            fname = self.save_transcript()
            print(f"Saved to {fname}")
            return "continue"
        if cmd == ":help":
            self.print_help()
            return "continue"
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
                user_input = input("\n---\nYou> ").strip()
                if not user_input:
                    continue

                action = self.handle_command(user_input)
                if action == "break":
                    break
                if action == "continue":
                    continue

                self.messages.append(
                    {"role": "user", "content": user_input})
                prompt = self.trim_to_prompt_budget()

                print("\n---\nAgent> ", end="", flush=True)
                response = self.generate(prompt)
                self.messages.append(
                    {"role": "assistant", "content": response})

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    session = QwenNPU()
    session.setup()
    session.run()