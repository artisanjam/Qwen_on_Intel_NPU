"""Trim, guard, gate, stripping and transcript checks (stubbed deps)."""
import io
import os
import sys
import tempfile
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
from qwen_npu.watchdog import extract_thinking, strip_thinking

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

# 10. closed think block yields its thinking
check("thinking extracted from closed block",
      extract_thinking("<think> mulling over the question </think>"
                       "Here is the answer.") == "mulling over the question")

# 11. interrupted turn: everything after the opener is thinking
check("unclosed opener yields thinking",
      extract_thinking("<think> mulling over the question")
      == "mulling over the question")

# 12. ordinary reply has no thinking
check("plain reply yields no thinking",
      extract_thinking("no thinking here.") == "")

# 13. record_answer keeps thinking for the transcript only;
# saved chat shows the thought without think tags
session = new_session([])
session.messages.append({"role": "user", "content": "hello there"})
session.record_answer("<think> mulling over the question </think>"
                      "Here is the answer.")
entry = session.messages[-1]
check("history content stays stripped",
      entry["content"] == "Here is the answer.")
check("thinking retained for transcript",
      entry.get("thinking") == "mulling over the question")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
    os.chdir(tmpdir)
    fname = session.save_transcript()
    with open(fname, encoding="utf-8") as handle:
        saved = handle.read()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
check("transcript shows thought and reply",
      "Assistant thought:\nmulling over the question" in saved
      and "Assistant: Here is the answer." in saved)
check("transcript carries no think tags", "<think" not in saved)

# 14. a sentence formulated in thought, then spoken verbatim, is
# not a loop
phrase = ("Hey there! I'm doing well, thanks for asking. It's been "
          "a great day so far--what have you been up to?")
session = new_session(["<think> ", phrase, "</think>", " ", phrase,
                       " Sure about that."])
session.messages.append({"role": "user", "content": "hello there"})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    response = session.generate("prompt")
check("thought-to-speech repeat passes",
      "[stopped: repeated phrasing]" not in out.getvalue())
check("full reply kept", len(response) == len("<think> ") +
      2 * len(phrase) + len("</think>") + len(" ") +
      len(" Sure about that."))

# 15. repetition inside the thinking channel still stops
tokens = ["<think> ", long[:30], long[30:] + ". ", " mulling over it. ",
          long[:30], long[30:] + ". "]
session = new_session(tokens)
session.messages.append({"role": "user", "content": "hello there"})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    session.generate("prompt")
check("within-think repeat stops",
      "[stopped: repeated phrasing]" in out.getvalue())

# 16. a prior sentence formulated in thought and then spoken once
# is one quote, not two replays
session = new_session(["<think> ", s1[:30], s1[30:] + ". ", "</think>",
                       " ", s1[:30], s1[30:] + ". ", " and fresh words"])
session.messages.append({"role": "assistant", "content": prior})
session.messages.append({"role": "user", "content": "go on"})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    session.generate("prompt")
check("think-then-speech prior quote passes",
      "[stopped: repeated phrasing]" not in out.getvalue())

print("failures:", failures)
sys.exit(1 if failures else 0)
