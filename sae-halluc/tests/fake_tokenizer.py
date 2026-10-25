"""Tiny stand-in tokenizers with HuggingFace-style offsets, for testing alignment logic
without transformers. Modes mimic the two offset conventions seen in the wild and a
sub-word mode that forces tokens to straddle span edges.
"""

import re
import zlib

_WORD = re.compile(r"\s*\w+|\s*[^\w\s]|\s+")


class FakeTokenizer:
    """mode:
        'word_untrimmed' : word / punctuation tokens, leading space INSIDE the offsets
        'word_trimmed'   : same tokens, offsets exclude the leading space
        'chunk'          : words split into pieces of `chunk` characters (tokens straddle spans)
    """

    def __init__(self, mode="word_untrimmed", chunk=3, bos=True):
        self.mode = mode
        self.chunk = chunk
        self.bos = bos

    def _pieces(self, text):
        out = []
        for m in _WORD.finditer(text):
            s, e = m.start(), m.end()
            tok = text[s:e]
            if self.mode == "chunk":
                lead = len(tok) - len(tok.lstrip())
                body_start = s + lead
                if tok.strip() and tok.strip()[0].isalnum() and len(tok.strip()) > self.chunk:
                    pos = body_start
                    first = True
                    while pos < e:
                        piece_end = min(pos + self.chunk, e)
                        out.append((s if first else pos, piece_end))
                        first = False
                        pos = piece_end
                    continue
            out.append((s, e))
        if self.mode == "word_trimmed":
            trimmed = []
            for s, e in out:
                while s < e and text[s].isspace():
                    s += 1
                trimmed.append((s, e))
            return trimmed
        return out

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
        pieces = self._pieces(text)
        ids = [zlib.crc32(text[s:e].encode()) % 100000 + 10 for s, e in pieces]
        offsets = list(pieces)
        if add_special_tokens and self.bos:
            ids = [1] + ids
            offsets = [(0, 0)] + offsets
        return {"input_ids": ids, "offset_mapping": offsets}
