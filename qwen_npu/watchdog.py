"""Loop detection and think-block handling for streamed text.

The guard ends a turn that repeats itself verbatim; stripping
keeps thinking out of stored history, and extraction keeps it for
the saved transcript.
"""

import re

from . import config

SENTENCE_END = re.compile(r"[.!?]")
THINK_BLOCK = re.compile(r"<think.*?</think>\s*", re.DOTALL)
THINK_CAPTURE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def extract_thinking(text):
    """Thinking only: the inside of closed think blocks.

    A turn interrupted mid-thought leaves an unclosed opener; if no
    block was ever closed, everything after the opener is thinking.
    A reply with no thinking yields "".
    """
    text = text or ""
    blocks = [m.group(1).strip() for m in THINK_CAPTURE.finditer(text)]
    if blocks:
        return "\n\n".join(blocks)
    if THINK_OPEN in text:
        return text.split(THINK_OPEN, 1)[1].strip()
    return ""


def strip_thinking(text):
    """Visible reply only; thinking stays out of model history.

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
    """End a turn that repeats itself verbatim within one channel.

    Thinking and speech are tracked separately: formulating a
    sentence in thought and then speaking it is normal dialogue,
    while repetition inside one channel is the loop signature.
    Cross-turn: distinct sentences replayed from the previous
    answer are counted -- quoting is legitimate once, so
    WATCH_PRIOR_MATCHES distinct replays are needed to call it a
    loop.
    """

    def __init__(self, prior):
        self.prior = set(sentences(prior))
        self.seen = {"think": set(), "speech": set()}
        self.prior_seen = set()
        self.buffer = ""
        self.in_think = False

    def _channel(self, part):
        """Channel a completed sentence was emitted in.

        Tags inside a part split it across channels; the sentence
        is assigned to whichever side of the tag holds most of its
        text. A tag with no sentence around it only moves the
        stream state.
        """
        start = self.in_think
        open_idx = part.find(THINK_OPEN)
        close_idx = part.find(THINK_CLOSE)
        for idx, tag in sorted([(open_idx, THINK_OPEN),
                                (close_idx, THINK_CLOSE)]):
            if idx >= 0:
                self.in_think = tag == THINK_OPEN
        if close_idx >= 0:
            before = len(part[:close_idx])
            after = len(part[close_idx + len(THINK_CLOSE):])
            if before >= after:
                return "think" if start else "speech"
            return "speech"
        if open_idx >= 0:
            before = len(part[:open_idx])
            after = len(part[open_idx + len(THINK_OPEN):])
            if before > after:
                return "think" if start else "speech"
            return "think"
        return "think" if start else "speech"

    def observe(self, token):
        """True once the turn should stop."""
        self.buffer += token
        parts = SENTENCE_END.split(self.buffer)
        self.buffer = parts[-1]  # keep the unterminated tail
        for part in parts[:-1]:
            sentence = normalize(
                part.replace(THINK_OPEN, "").replace(THINK_CLOSE, ""))
            seen = self.seen[self._channel(part)]
            if len(sentence) < config.WATCH_MIN_CHARS:
                continue
            if sentence in seen:
                return True
            if sentence in self.prior:
                self.prior_seen.add(sentence)
                if len(self.prior_seen) >= config.WATCH_PRIOR_MATCHES:
                    return True
            seen.add(sentence)
        return False
