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


def train(a, b, y, out_dir, epochs=1, batch=256, lr=2e-5, warmup=0.05, log=print, base=BASE,
          eval_data=None, eval_every=2000):
    """Fine-tune ``base`` as a pair classifier.

    With several epochs, each epoch's weights are saved to ``out_dir/epoch<k>``.
    If ``eval_data`` = (a, b, y) is given, the model is evaluated on it every
    ``eval_every`` steps and at each epoch end (log-loss, AUC), and the best
    checkpoint by log-loss is also saved to ``out_dir/best``.
    """
    from sklearn.metrics import log_loss, roc_auc_score
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).cuda()
    dl = DataLoader(_Pairs(a, b, y), batch_size=batch, shuffle=True, num_workers=8,
                    collate_fn=_collate(tok), drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup * total)) * max(0.0, (total - s) / max(1, total * (1 - warmup))))
    lossf = torch.nn.BCEWithLogitsLoss()
    best = {"logloss": float("inf")}

    def evaluate(tag):
        if eval_data is None:
            return
        model.eval()
        p = _predict(model, tok, eval_data[0], eval_data[1])
        model.train()
        ll, auc = log_loss(eval_data[2], np.clip(p, 1e-6, 1 - 1e-6)), roc_auc_score(eval_data[2], p)
        log(f"eval {tag}: logloss {ll:.4f} auc {auc:.4f}")
        if ll < best["logloss"]:
            best.update(logloss=ll, auc=auc, at=tag)
            model.save_pretrained(f"{out_dir}/best")
            tok.save_pretrained(f"{out_dir}/best")

    model.train()
    step, t0, run = 0, time.time(), 0.0
    for ep in range(1, epochs + 1):
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
                log(f"epoch {ep} step {step}/{total} loss {run:.4f} | {rate:,.0f} pairs/s | eta {(total - step) * batch / rate / 60:.0f} min")
            if eval_every and step % eval_every == 0:
                evaluate(f"step {step}")
        d = out_dir if epochs == 1 else f"{out_dir}/epoch{ep}"
        model.save_pretrained(d)
        tok.save_pretrained(d)
        evaluate(f"end of epoch {ep}")
    if eval_data is not None:
        log(f"best checkpoint: {best}")
    return out_dir


@torch.no_grad()
def _predict(model, tok, a, b, batch=1024):
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


@torch.no_grad()
def score(a, b, model_dir, batch=1024):
    """Match probability for each (a[i], b[i]); pairs are length-sorted for speed."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    return _predict(model, tok, a, b, batch)
