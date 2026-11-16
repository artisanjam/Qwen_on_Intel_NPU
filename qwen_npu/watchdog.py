"""Loop detection and think-block stripping for streamed text.

The guard ends a turn that repeats itself verbatim; stripping
keeps thinking out of stored history.
"""

import re

from . import config

SENTENCE_END = re.compile(r"[.!?]")
THINK_BLOCK = re.compile(r"<think.*?</think>\s*", re.DOTALL)
THINK_OPEN = "<think>"


def strip_thinking(text):
    """Visible reply only; thinking is shown live but not stored.

    A turn interrupted mid-thought leaves an unclosed block, and
    anything after the opener is thinking rather than answer.
    """
    text = THINK_BLOCK.sub("", text or "")
    return text.split(THINK_OPEN, 1)[0].strip()


def normalize(part):
    """Lowercase, whitespace collapsed: casing is not a loop signal."""
    return " ".join(part.lower().split())


def sentences(text):
    """Normalized sentences long enough to be loop material."""
    return [s for s in map(normalize, SENTENCE_END.split(text or ""))
            if len(s) >= config.WATCH_MIN_CHARS]


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
            if len(sentence) < config.WATCH_MIN_CHARS:
                continue
            if sentence in self.seen:
                return True
            if sentence in self.prior:
                self.prior_hits += 1
                if self.prior_hits >= config.WATCH_PRIOR_MATCHES:
                    return True
            self.seen.add(sentence)
        return False
