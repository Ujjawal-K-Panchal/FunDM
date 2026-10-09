"""The option heads and their loss, and how a question and its options become one row for them."""
from dataclasses import dataclass
import math
import random

import torch
from torch import nn


class OptionHead(nn.Module):
    """q = W_q·h[answer], k_i = W_k·h[sep_i]; logits q·k_i / sqrt(dim), -inf on padding slots."""

    def __init__(self, hidden: int, dim: int = 128):
        super().__init__()
        self.w_q = nn.Linear(hidden, dim, bias=False)
        self.w_k = nn.Linear(hidden, dim, bias=False)
        self.dim = dim

    def forward(self, h: torch.Tensor, answer_pos: torch.Tensor, sep_pos: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        :param h: (B, T, H) final hidden states of the frozen LM.
        :param answer_pos: (B,) query position per row.
        :param sep_pos: (B, K) key positions; padding slots may hold any index.
        :param mask: (B, K) True for real options.
        :return: (B, K) logits.
        """
        rows = torch.arange(h.size(0), device=h.device)
        return self.score(h[rows, answer_pos], h[rows[:, None], sep_pos.masked_fill(~mask, 0)], mask)

    def score(self, hq: torch.Tensor, hk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Logits from gathered states: hq (B, H) at the answer, hk (B, K, H) at the separators."""
        q, k = self.w_q(hq.float()), self.w_k(hk.float())
        logits = torch.einsum("bd,bkd->bk", q, k) / math.sqrt(self.dim)
        return logits.masked_fill(~mask, float("-inf"))


class ComparatorHead(nn.Module):
    """One bidirectional transformer layer over the K option states and the query, then q·k scoring."""

    def __init__(self, hidden: int, dim: int = 128, width: int = 512, heads: int = 8, ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(hidden, width)
        self.kind = nn.Embedding(2, width)          # 0 = option, 1 = query; no positions, so no order signal is added
        self.layer = nn.TransformerEncoderLayer(width, heads, ff, dropout, activation="gelu",
                                                batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(width)
        self.w_q = nn.Linear(width, dim, bias=False)
        self.w_k = nn.Linear(width, dim, bias=False)
        self.dim = dim

    def score(self, hq: torch.Tensor, hk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.inp(torch.cat([hk.float(), hq.float()[:, None]], 1))
        kind = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        kind[:, -1] = 1
        pad = torch.cat([~mask, torch.zeros_like(mask[:, :1])], 1)
        x = self.norm(self.layer(x + self.kind(kind), src_key_padding_mask=pad))
        q, k = self.w_q(x[:, -1]), self.w_k(x[:, :-1])
        return (torch.einsum("bd,bkd->bk", q, k) / math.sqrt(self.dim)).masked_fill(~mask, float("-inf"))


class CLMStyleHead(nn.Module):
    """CLM's head shape: two MLPs (hidden->1536->1536->512, GELU, LayerNorm), cosine x learned scale capped at 100."""

    def __init__(self, hidden: int, width: int = 1536, proj: int = 512):
        super().__init__()
        def mlp():
            return nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width),
                                 nn.GELU(), nn.Linear(width, proj))
        self.state, self.action = mlp(), mlp()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def score(self, hq: torch.Tensor, hk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        zs = nn.functional.normalize(self.state(hq.float()), dim=-1)
        za = nn.functional.normalize(self.action(hk.float()), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100.0) * torch.einsum("bd,bkd->bk", zs, za)
        return logits.masked_fill(~mask, float("-inf"))


HEADS = {"linear": OptionHead, "comparator": ComparatorHead, "clm_mlp": CLMStyleHead}


def soft_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against targets / sum(targets); every row needs a positive target."""
    t = targets * mask
    t = t / t.sum(-1, keepdim=True)
    logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    return -(t * logp).sum(-1).mean()


# Rows read together: a question and its options in one causal row, plus the positions the head reads.
SEP = " ::"          # closes each option; one Qwen3 token (3504)
ANSWER = " answer:"  # its last token is the head's query


@dataclass
class OptionRow:
    ids: list[int]
    answer_pos: int       # query position: last token of ANSWER
    sep_pos: list[int]    # key positions: the separator after each option, in row order
    order: list[int]      # row slot -> index into the caller's options


def build_row(tok, question: str, options: list[str], rng: random.Random | None = None) -> OptionRow:
    """Tokenise piece by piece, so recorded positions stay exact even if an option contains "::"."""
    order = list(range(len(options)))
    if rng is not None:
        rng.shuffle(order)
    def enc(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)
    sep = enc(SEP)
    ids = enc(question.rstrip() + " rubric:")
    sep_pos: list[int] = []
    for i in order:
        ids += enc(" " + options[i].strip()) + sep
        sep_pos.append(len(ids) - 1)
    ids += enc(ANSWER)
    return OptionRow(ids, len(ids) - 1, sep_pos, order)


def collate(rows: list[OptionRow], pad_id: int, device: torch.device | str = "cpu"):
    """Right-pad ids to T and key slots to K. -> ids, attention, answer_pos, sep_pos, key_mask."""
    T = max(len(r.ids) for r in rows)
    K = max(len(r.sep_pos) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), T), dtype=torch.long)
    sep = torch.zeros((len(rows), K), dtype=torch.long)
    mask = torch.zeros((len(rows), K), dtype=torch.bool)
    for b, r in enumerate(rows):
        ids[b, :len(r.ids)] = torch.tensor(r.ids)
        attn[b, :len(r.ids)] = 1
        sep[b, :len(r.sep_pos)] = torch.tensor(r.sep_pos)
        mask[b, :len(r.sep_pos)] = True
    answer = torch.tensor([r.answer_pos for r in rows])
    return ids.to(device), attn.to(device), answer.to(device), sep.to(device), mask.to(device)


def targets_in_row_order(rows: list[OptionRow], labels: list[list[float]], K: int) -> torch.Tensor:
    """(B, K) targets, permuted to each row's option order; padding slots are 0."""
    t = torch.zeros((len(rows), K))
    for b, (r, lab) in enumerate(zip(rows, labels, strict=True)):
        assert len(lab) == len(r.order), "one label per option"
        t[b, :len(r.order)] = torch.tensor([float(lab[i]) for i in r.order])
    return t
