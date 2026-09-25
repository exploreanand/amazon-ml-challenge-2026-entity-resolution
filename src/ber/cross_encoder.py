"""Cross-encoder pair classifier: both records in one input, fine-tuned end to end.

Base model: intfloat/multilingual-e5-base (MIT licence, 278M params), with a
single-logit classification head. It sees the raw (HTML-unescaped) name and
address of both records, so it can learn which small differences separate a
noisy true match from a decoy.
"""

import html
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "intfloat/multilingual-e5-base"
MAX_LEN = 96


def record_text(names, addresses):
    return [f"{html.unescape(n)} | {html.unescape(a)}" for n, a in zip(names, addresses)]


class _Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def _collate(tok):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def train(a, b, y, out_dir, epochs=1, batch=256, lr=2e-5, warmup=0.05, log=print, base=BASE):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).cuda()
    dl = DataLoader(_Pairs(a, b, y), batch_size=batch, shuffle=True, num_workers=8,
                    collate_fn=_collate(tok), drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup * total)) * max(0.0, (total - s) / max(1, total * (1 - warmup))))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0, run = 0, time.time(), 0.0
    for _ in range(epochs):
        for enc in dl:
            yb = enc.pop("labels").cuda(non_blocking=True)
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0:
                rate = step * batch / (time.time() - t0)
                log(f"step {step}/{total} loss {run:.4f} | {rate:,.0f} pairs/s | eta {(total - step) * batch / rate / 60:.0f} min")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


@torch.no_grad()
def score(a, b, model_dir, batch=1024):
    """Match probability for each (a[i], b[i]); pairs are length-sorted for speed."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), batch):
        idx = order[s:s + batch]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1).float()
        out[idx] = torch.sigmoid(logit).cpu().numpy()
    return out
