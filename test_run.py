"""Integration tests for trim, guard, gate and stripping
(stubbed deps)."""
import io
import os
import sys
import types
import contextlib


class FakeShapeTensor:
    def __init__(self, n):
        self.shape = [1, n]


class FakeTokenized:
    def __init__(self, n):
        self.input_ids = FakeShapeTensor(n)


class FakeTokenizer:
    def encode(self, prompt):
        return FakeTokenized(len(prompt) // 3)


class FakeTemplate:
    def render(self, messages, add_generation_prompt, enable_thinking):
        return " ".join(m["content"] for m in messages)


class FakeEnv:
    def from_string(self, source):
        return FakeTemplate()


class GenerationConfig:
    pass


def make_pipe(tokens):
    class Pipe:
        def __init__(self, *args, **kwargs):
            pass

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompt, config, streamer):
            for token in tokens:
                if streamer(token):
                    return
    return Pipe()


ov = types.ModuleType("openvino_genai")
ov.GenerationConfig = GenerationConfig
ov.LLMPipeline = make_pipe([])
sys.modules["openvino_genai"] = ov

jinja2 = types.ModuleType("jinja2")
jinja2.Environment = FakeEnv
sys.modules["jinja2"] = jinja2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from qwen_npu.config import PROMPT_BUDGET
from qwen_npu.session import QwenNPU
from qwen_npu.watchdog import strip_thinking

failures = []


def check(name, ok):
    print(("PASS" if ok else "FAIL"), name)
    if not ok:
        failures.append(name)


def new_session(tokens):
    session = QwenNPU()
    session.pipe = make_pipe(tokens)
    session.tokenizer = FakeTokenizer()
    return session


# 1. trim: exact token counts, pairs removed until under budget
session = new_session([])
for i in range(8):
    session.messages.append({"role": "user", "content": "u" * 1500})
    session.messages.append({"role": "assistant", "content": "a" * 1500})
prompt = session.trim_to_prompt_budget()
check("trim under budget", len(prompt) // 3 <= PROMPT_BUDGET)
check("trim keeps recent pair",
      session.messages[-2]["content"] == "u" * 1500)

# 2. within-turn verbatim repeat stops the turn
long = ("You're talking about a way of being in the world, how you "
        "engage with the unknown")
tokens = [long[:30], long[30:] + ". ", "Sure. ", long[:30], long[30:] + ". "]
session = new_session(tokens)
session.messages.append({"role": "user", "content": "hello there"})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    response = session.generate("prompt")
check("within-turn stop fires", "[stopped: repeated phrasing]" in out.getvalue())
check("partial kept", len(response) > len(long) + 6)

# 3. cross-turn replay: two prior sentences re-emitted -> stop
s1 = "That's the risk and beauty, the part of the game and art you " \
     "are already living through"
s2 = "You said the most important line about scratching at the " \
     "mundane and mystery dichotomy"
prior = s1 + ". " + s2 + "."
session = new_session([s1[:30], s1[30:] + ". ", "Fair. ", s2 + ". ",
                       "extra words"])
session.messages.append({"role": "assistant", "content": prior})
session.messages.append({"role": "user", "content": "go on"})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    session.generate("prompt")
check("cross-turn replay stops", "[stopped: repeated phrasing]" in out.getvalue())

# 4. single legitimate echo does not stop
session = new_session([s2[:30], s2[30:] + ". ",
                       "and fresh material follows here"])
session.messages.append({"role": "assistant", "content": prior})
session.messages.append({"role": "user", "content": "go on"})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    response = session.generate("prompt")
check("one echo passes", "[stopped: repeated phrasing]" not in out.getvalue())

# 5. a turn that alone outgrows the budget is not sendable
session = new_session([])
session.messages.append({"role": "user",
                         "content": "x" * (PROMPT_BUDGET * 3 + 9)})
prompt = session.trim_to_prompt_budget()
check("oversized turn declined", not session.sendable(prompt))

# 6. an ordinary turn is sendable
session = new_session([])
session.messages.append({"role": "user", "content": "hello there"})
check("ordinary turn sendable", session.sendable(session.render_prompt()))

# 7. closed think block is stripped from the stored reply
raw = "<think> mulling over the question </think>Here is the answer."
check("closed think stripped", strip_thinking(raw) == "Here is the answer.")

# 8. interrupted turn: unclosed opener drops everything after it
raw = "<think> mulling over the question"
check("unclosed think drops tail", strip_thinking(raw) == "")

# 9. ordinary reply passes through untouched
check("plain reply untouched",
      strip_thinking("no thinking here.") == "no thinking here.")

print("failures:", failures)
sys.exit(1 if failures else 0)
