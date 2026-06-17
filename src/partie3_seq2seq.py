"""
Partie III — RNN / LSTM / GRU et Seq2Seq sur corpus parallèle EN-FR (ManyThings).

Corpus réel téléchargé automatiquement (équivalent fra-eng du sujet EMSI).
Compare RNN/LSTM/GRU, Seq2Seq, décodage glouton / beam, BLEU.
"""
from __future__ import annotations

import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer

from .corpus import load_parallel_pairs
from .vocab_utils import build_vocab
from .experiment_io import append_global_summary, save_figure, save_metrics, save_table

SEED = 42
MAX_LEN = 25
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 256
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, tokenizer, max_len=MAX_LEN):
        self.samples = []
        for src, tgt in pairs:
            src_ids = [src_vocab[t] for t in tokenizer(src)][: max_len - 2]
            tgt_ids = [tgt_vocab[t] for t in tokenizer(tgt)][: max_len - 2]
            self.samples.append(
                (torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RecurrentLM(nn.Module):
    def __init__(self, vocab_size, pad_idx, cell="gru", hidden=HIDDEN_DIM, embed=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=pad_idx)
        if cell == "rnn":
            self.rnn = nn.RNN(embed, hidden, batch_first=True)
        elif cell == "lstm":
            self.rnn = nn.LSTM(embed, hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        self.cell = cell

    def forward(self, x):
        out, _ = self.rnn(self.embed(x))
        return self.fc(out)


class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed=EMBED_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, src):
        _, hidden = self.rnn(self.embed(src))
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed=EMBED_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, tgt_in, hidden):
        out, hidden = self.rnn(self.embed(tgt_in), hidden)
        return self.fc(out), hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt_in):
        hidden = self.encoder(src)
        logits, _ = self.decoder(tgt_in, hidden)
        return logits


def masked_nll(logits, targets, pad_idx):
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    n_tokens = (targets != pad_idx).sum().item()
    return loss / max(n_tokens, 1), n_tokens


def run(export: bool = True, lm_epochs: int = 3, s2s_epochs: int = 5) -> dict:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer("basic_english")

    train_pairs, val_pairs = load_parallel_pairs(max_train=8000, max_val=800)

    src_vocab = build_vocab(tokenizer, train_pairs, lang_idx=0, specials=SPECIALS)
    tgt_vocab = build_vocab(tokenizer, train_pairs, lang_idx=1, specials=SPECIALS)
    pad_src, pad_tgt = src_vocab["<pad>"], tgt_vocab["<pad>"]
    bos_tgt, eos_tgt = tgt_vocab["<bos>"], tgt_vocab["<eos>"]

    def collate(batch):
        src_list, tgt_list = zip(*batch)
        src_pad = pad_sequence(src_list, batch_first=True, padding_value=pad_src)
        tgt_in = [torch.tensor([bos_tgt] + t.tolist(), dtype=torch.long) for t in tgt_list]
        tgt_out = [torch.tensor(t.tolist() + [eos_tgt], dtype=torch.long) for t in tgt_list]
        return (
            src_pad,
            pad_sequence(tgt_in, batch_first=True, padding_value=pad_tgt),
            pad_sequence(tgt_out, batch_first=True, padding_value=pad_tgt),
        )

    train_dl = DataLoader(
        TranslationDataset(train_pairs, src_vocab, tgt_vocab, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )

    lm_results = {}
    for cell in ["rnn", "lstm", "gru"]:
        model = RecurrentLM(len(src_vocab), pad_src, cell=cell).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(lm_epochs):
            model.train()
            total_loss, n_tok = 0.0, 0
            for src, _, _ in train_dl:
                src = src.to(device)
                inp, tgt = src[:, :-1], src[:, 1:]
                logits = model(inp)
                loss, nt = masked_nll(logits, tgt, pad_src)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * nt
                n_tok += nt
            ppl = math.exp(total_loss / n_tok)
            print(f"  LM {cell} epoch {ep+1}/{lm_epochs} perplexite={ppl:.2f}")
        lm_results[cell] = ppl

    model = Seq2Seq(
        Encoder(len(src_vocab), pad_src), Decoder(len(tgt_vocab), pad_tgt)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    s2s_losses = []
    for ep in range(s2s_epochs):
        model.train()
        total, n_tok = 0.0, 0
        t0 = time.time()
        for src, tgt_in, tgt_out in train_dl:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss, nt = masked_nll(logits, tgt_out, pad_tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * nt
            n_tok += nt
        s2s_losses.append(total / n_tok)
        print(f"  Seq2Seq epoch {ep+1}/{s2s_epochs} loss/token={s2s_losses[-1]:.4f} ({time.time()-t0:.0f}s)")

    def translate(sentence, method="greedy", beam_width=3):
        model.eval()
        src = torch.tensor([[src_vocab[t] for t in tokenizer(sentence)]], device=device)
        hidden = model.encoder(src)
        if method == "greedy":
            dec_in = torch.tensor([[bos_tgt]], device=device)
            generated = [bos_tgt]
            for _ in range(MAX_LEN):
                logits, hidden = model.decoder(dec_in, hidden)
                nid = logits[:, -1, :].argmax(-1).item()
                generated.append(nid)
                if nid == eos_tgt:
                    break
                dec_in = torch.tensor([[nid]], device=device)
        else:
            beams = [([bos_tgt], hidden, 0.0)]
            for _ in range(MAX_LEN):
                new_beams = []
                for seq, h, score in beams:
                    if seq[-1] == eos_tgt:
                        new_beams.append((seq, h, score))
                        continue
                    dec_in = torch.tensor([[seq[-1]]], device=device)
                    logits, new_h = model.decoder(dec_in, h)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                    topk = log_probs.topk(beam_width)
                    for i in range(topk.values.size(0)):
                        nid = topk.indices[i].item()
                        new_beams.append((seq + [nid], new_h, score + topk.values[i].item()))
                beams = sorted(new_beams, key=lambda x: x[2] / len(x[0]), reverse=True)[:beam_width]
            generated = beams[0][0] if beams else [bos_tgt]
        itos = tgt_vocab.get_itos()
        words = []
        for i in generated:
            w = itos[i]
            if w == "<eos>":
                break
            if w not in ("<pad>", "<bos>"):
                words.append(w)
        return " ".join(words)

    try:
        import sacrebleu

        def bleu_score(n=80, method="greedy"):
            hyps, refs = [], []
            for src, tgt in val_pairs[:n]:
                hyps.append(translate(src, method=method))
                refs.append([" ".join(tokenizer(tgt))])
            return sacrebleu.corpus_bleu(hyps, refs).score
    except ImportError:
        def bleu_score(n=80, method="greedy"):
            return 0.0

    bleu_greedy = bleu_score(60, "greedy")
    bleu_beam = bleu_score(60, "beam")
    print(f"  BLEU greedy={bleu_greedy:.2f} beam={bleu_beam:.2f}")

    out = {
        "device": str(device),
        "corpus": "fra-eng (ManyThings)",
        "perplexite_finale": lm_results,
        "seq2seq_loss_finale": s2s_losses[-1],
        "bleu_greedy": bleu_greedy,
        "bleu_beam": bleu_beam,
    }

    if export:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(lm_results.keys()), list(lm_results.values()))
        ax.set_ylabel("Perplexite")
        ax.set_title("RNN vs LSTM vs GRU")
        save_figure(fig, "partie3", "perplexite_rnn_lstm_gru.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(s2s_losses)
        ax.set_title("Perte Seq2Seq")
        ax.set_xlabel("Epoque")
        save_figure(fig, "partie3", "courbe_seq2seq_loss.png")

        save_table("partie3", lm_results, "perplexite_comparaison.csv")
        save_metrics("partie3", out)
        append_global_summary({"partie": "III_Seq2Seq", **out})

    return out
