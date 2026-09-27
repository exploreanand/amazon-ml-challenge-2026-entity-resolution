"""Qwen3-Reranker fine-tuned as a pair classifier (v11 experiment).

Qwen/Qwen3-Reranker-0.6B (Apache-2.0) is a causal LM that answers "yes" or "no"
to whether a document fits a query. The pair score here is logit("yes") -
logit("no") at the last position of the model card's prompt, with the two
records as query and document. Fine-tuning that score (BCE) keeps the pretrained
judgement instead of starting from a new classification head. Checkpoints carry
MARKER, so ber.cross_encoder.score dispatches to this module.
"""

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-Reranker-0.6B"
MARKER = "qwen_rerank.json"
MAX_CHARS = 160  # per record; the prompt itself is never truncated
INSTRUCT = "Given two business records, judge whether they refer to the same business"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
          'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompt(a, b):
    return f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {a[:MAX_CHARS]}\n<Document>: {b[:MAX_CHARS]}{SUFFIX}"


class _Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return prompt(self.a[i], self.b[i]), (self.y[i] if self.y is not None else 0.0)


def _load(path, dtype):
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda()
    return tok, model, tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")


def _yes_no(model, enc, yes, no):
    """logit(yes) - logit(no) at the last position (left padding), without the full-vocabulary head."""
    h = model.model(**enc).last_hidden_state[:, -1]
    z = h @ model.lm_head.weight[[yes, no]].T
    return (z[:, 0] - z[:, 1]).float()


@torch.no_grad()
def _predict(model, tok, yes, no, a, b, batch=256):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), batch):
        idx = order[s:s + batch]
        enc = tok([prompt(a[i], b[i]) for i in idx], padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = torch.sigmoid(_yes_no(model, enc, yes, no)).cpu().numpy()
    return out


def _save(model, tok, path):
    model.save_pretrained(path)
    tok.save_pretrained(path)
    json.dump({"arch": "qwen_rerank", "base": BASE}, open(f"{path}/{MARKER}", "w"))


def train(a, b, y, out_dir, epochs=1, batch=64, lr=1e-5, warmup=0.05, log=print, base=BASE,
          eval_data=None, eval_every=8000):
    """Same interface as ber.cross_encoder.train; checkpoints go to out_dir/epoch<k> and out_dir/best."""
    from sklearn.metrics import log_loss, roc_auc_score
    tok, model, yes, no = _load(base, torch.float32)

    def collate(items):
        x, t = zip(*items)
        enc = tok(list(x), padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(t, dtype=torch.float32)
        return enc

    dl = DataLoader(_Pairs(a, b, y), batch_size=batch, shuffle=True, num_workers=8, collate_fn=collate,
                    drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup * total)) * max(0.0, (total - s) / max(1, total * (1 - warmup))))
    lossf = torch.nn.BCEWithLogitsLoss()
    best = {"logloss": float("inf")}

    def evaluate(tag):
        if eval_data is None:
            return
        p = _predict(model, tok, yes, no, eval_data[0], eval_data[1])
        model.train()
        ll, auc = log_loss(eval_data[2], np.clip(p, 1e-6, 1 - 1e-6)), roc_auc_score(eval_data[2], p)
        log(f"eval {tag}: logloss {ll:.4f} auc {auc:.4f}")
        if ll < best["logloss"]:
            best.update(logloss=ll, auc=auc, at=tag)
            _save(model, tok, f"{out_dir}/best")

    model.train()
    step, t0, run = 0, time.time(), 0.0
    for ep in range(1, epochs + 1):
        for enc in dl:
            yb = enc.pop("labels").cuda(non_blocking=True)
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = _yes_no(model, enc, yes, no)
            loss = lossf(logit, yb)
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
        _save(model, tok, f"{out_dir}/epoch{ep}")
        evaluate(f"end of epoch {ep}")
    if eval_data is not None:
        log(f"best checkpoint: {best}")
    return out_dir


def score(a, b, model_dir, batch=256):
    tok, model, yes, no = _load(model_dir, torch.bfloat16)
    return _predict(model, tok, yes, no, a, b, batch)
