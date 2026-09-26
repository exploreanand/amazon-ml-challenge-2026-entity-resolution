"""Decoder-LLM pair classifier (Qwen3, Apache-2.0) fine-tuned with LoRA.

The two records go into one prompt; a sequence-classification head on the
last token outputs one match logit. Only LoRA adapters and the head are
trained, so a 1.7B model fits comfortably in GPU memory in bf16.
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128


def prompt(a, b):
    return f"Are these the same business?\nRecord A: {a}\nRecord B: {b}\nAnswer:"


class _Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return prompt(self.a[i], self.b[i]), (self.y[i] if self.y is not None else 0.0)


def _collate(tok):
    def f(batch):
        x, y = zip(*batch)
        enc = tok(list(x), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def load_model(base, lora_dir=None, r=16):
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"  # classification head reads the last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
    else:
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="SEQ_CLS",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    return tok, model.cuda()


@torch.no_grad()
def _predict(model, tok, a, b, batch=512):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), batch):
        idx = order[s:s + batch]
        enc = tok([prompt(a[i], b[i]) for i in idx], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1).float()
        out[idx] = torch.sigmoid(logit).cpu().numpy()
    return out


def train(a, b, y, out_dir, base, epochs=1, batch=64, lr=1e-4, warmup=0.03, log=print,
          eval_data=None, eval_every=2000):
    from sklearn.metrics import log_loss, roc_auc_score
    tok, model = load_model(base)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params: {sum(p.numel() for p in params):,}")
    dl = DataLoader(_Pairs(a, b, y), batch_size=batch, shuffle=True, num_workers=8,
                    collate_fn=_collate(tok), drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup * total)) * max(0.0, (total - s) / max(1, total * (1 - warmup))))
    lossf = torch.nn.BCEWithLogitsLoss()
    best = {"logloss": float("inf")}

    def evaluate(tag):
        if eval_data is None:
            return
        p = _predict(model, tok, eval_data[0], eval_data[1])
        model.train()
        ll, auc = log_loss(eval_data[2], np.clip(p, 1e-6, 1 - 1e-6)), roc_auc_score(eval_data[2], p)
        log(f"eval {tag}: logloss {ll:.4f} auc {auc:.4f}")
        if ll < best["logloss"]:
            best.update(logloss=ll, auc=auc, at=tag)
            model.save_pretrained(f"{out_dir}/best")

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
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 200 == 0:
                rate = step * batch / (time.time() - t0)
                log(f"epoch {ep} step {step}/{total} loss {run:.4f} | {rate:,.0f} pairs/s | eta {(total - step) * batch / rate / 60:.0f} min")
            if eval_every and step % eval_every == 0:
                evaluate(f"step {step}")
        model.save_pretrained(f"{out_dir}/epoch{ep}")
        evaluate(f"end of epoch {ep}")
    log(f"best checkpoint: {best}")


def score(a, b, base, lora_dir, batch=512):
    tok, model = load_model(base, lora_dir)
    return _predict(model, tok, a, b, batch)
