"""Construction de vocabulaire sans API torchtext récente (compatible 0.6 / Py3.14)."""
from __future__ import annotations

from collections import Counter


class SimpleVocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def set_default_index(self, idx: int):
        self.default_index = idx

    def get_itos(self):
        return self.itos


def build_vocab(tokenizer, pairs, lang_idx: int, specials: list[str], min_freq: int = 2) -> SimpleVocab:
    counter: Counter = Counter()
    for src, tgt in pairs:
        text = src if lang_idx == 0 else tgt
        counter.update(tokenizer(text))
    itos = list(specials)
    for word, count in counter.most_common():
        if count >= min_freq and word not in itos:
            itos.append(word)
    vocab = SimpleVocab(itos)
    vocab.set_default_index(itos.index("<unk>"))
    return vocab
