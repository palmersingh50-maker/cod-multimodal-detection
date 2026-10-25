from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cod.data.dataset import CODDataset, build_samples
from cod.evaluation.evaluate import evaluate
from cod.models import CODModel
from cod.training.losses import CODLoss
from cod.utils import dump_json, get_device, hardware_info, seed_everything

LOWER_IS_BETTER = {"mae"}


def build_datasets(cfg: dict, splits=("train", "val")) -> dict[str, CODDataset]:
    d, seed = cfg["data"], cfg["experiment"]["seed"]
    use_depth = cfg["model"]["fusion"] != "none"
    return {s: CODDataset(build_samples(d, s, seed), "cod10k", d["image_size"], train=(s == "train"),
                          use_depth=use_depth, depth_cache=d["depth_cache"]) for s in splits}


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def amp_dtype(setting, device) -> torch.dtype | None:
    if setting in (False, "off", "fp32"):
        return None
    if setting == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[setting]


class Trainer:
    def __init__(self, cfg: dict, device: str = "auto", resume: bool = False):
        self.cfg, self.t = cfg, cfg["train"]
        self.device = get_device(device)
        seed_everything(cfg["experiment"]["seed"])
        self.dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = CODModel(cfg).to(self.device)
        self.loss_fn = CODLoss(**cfg["loss"])
        self.ds = build_datasets(cfg)
        bs, nw = self.t["batch_size"], cfg["data"]["num_workers"]
        self.train_dl = DataLoader(self.ds["train"], bs, shuffle=True, drop_last=True, num_workers=nw,
                                   persistent_workers=nw > 0, generator=torch.Generator().manual_seed(cfg["experiment"]["seed"]))
        self.steps_per_epoch = min(len(self.train_dl), self.t.get("max_steps_per_epoch") or 10**9)
        self.opt_steps_per_epoch = max(1, self.steps_per_epoch // self.t["accum_steps"])
        total = self.opt_steps_per_epoch * self.t["epochs"]
        groups = [{"params": [p for p in self.model.backbone.parameters()], "lr": self.t["lr_backbone"]},
                  {"params": self.model.head_parameters(), "lr": self.t["lr_head"]}]
        self.opt = torch.optim.AdamW(groups, weight_decay=self.t["weight_decay"])
        warm = self.opt_steps_per_epoch * self.t["warmup_epochs"]
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_lambda(s, warm, total))
        self.dtype = amp_dtype(self.t["amp"], self.device)
        self.scaler = None
        self.epoch, self.best, self.bad_epochs = 0, None, 0
        self.metric = self.t["val_metric"]
        self.sign = -1.0 if self.metric in LOWER_IS_BETTER else 1.0
        self.tb = SummaryWriter(str(self.dir / "tb"))
        self.history: list[dict] = []
        if resume and (self.dir / "last.pt").exists():
            self._load(self.dir / "last.pt")
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.info = {"hardware": hardware_info(), "device": str(self.device),
                     "params_total": sum(p.numel() for p in self.model.parameters()),
                     "params_backbone": sum(p.numel() for p in self.model.backbone.parameters()),
                     "n_train": len(self.ds["train"]), "n_val": len(self.ds["val"])}

    # ------------------------------------------------------------------ io
    def _state(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                "epoch": self.epoch, "best": self.best, "bad_epochs": self.bad_epochs, "history": self.history,
                "config": self.cfg, "rng": torch.get_rng_state()}

    def _load(self, path: Path) -> None:
        s = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.epoch, self.best, self.bad_epochs, self.history = s["epoch"], s["best"], s["bad_epochs"], s["history"]
        torch.set_rng_state(s["rng"])
        print(f"resumed from {path} at epoch {self.epoch}")

    def _set_backbone_trainable(self, epoch: int) -> None:
        frozen = epoch < self.t["backbone_freeze_epochs"]
        self.model.backbone.set_trainable(not frozen, self.t["backbone_trainable_blocks"])

    # ---------------------------------------------------------------- epoch
    def _train_epoch(self) -> dict:
        m, t = self.model, self.t
        m.train()
        self._set_backbone_trainable(self.epoch)
        agg: dict[str, list[float]] = {}
        bad = 0
        self.opt.zero_grad(set_to_none=True)
        it = tqdm(self.train_dl, total=self.steps_per_epoch, desc=f"epoch {self.epoch + 1}/{t['epochs']}", leave=False)
        for i, b in enumerate(it):
            if i >= self.steps_per_epoch:
                break
            img, mask = b["image"].to(self.device), b["mask"].to(self.device)
            depth = valid = None
            if "depth" in b:
                depth, valid = b["depth"].to(self.device), b["depth_valid"].to(self.device)
                if t["depth_dropout"] > 0:  # simulate missing/unreliable depth -> keeps RGB path strong
                    valid = valid * (torch.rand_like(valid) >= t["depth_dropout"]).float()
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
                out = m(img, depth, valid)
            loss, parts = self.loss_fn(out, mask)
            if not torch.isfinite(loss):
                bad += 1
                self.opt.zero_grad(set_to_none=True)
                print(f"WARNING non-finite loss at epoch {self.epoch + 1} step {i} ({bad} consecutive)")
                if bad >= t["max_bad_steps"]:
                    raise RuntimeError("training diverged: too many consecutive non-finite losses")
                continue
            bad = 0
            (loss / t["accum_steps"]).backward()
            if (i + 1) % t["accum_steps"] == 0:
                if t["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), t["grad_clip"])
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
            for k, v in {"loss": float(loss), **parts}.items():
                agg.setdefault(k, []).append(v)
            if i % 20 == 0:
                it.set_postfix(loss=f"{np.mean(agg['loss'][-20:]):.4f}")
        return {k: float(np.mean(v)) for k, v in agg.items()}

    # ----------------------------------------------------------------- loop
    def fit(self) -> dict:
        t0 = time.time()
        dump_json(self.info, self.dir / "env.json")
        for self.epoch in range(self.epoch, self.t["epochs"]):
            e0 = time.time()
            tr = self._train_epoch()
            va = evaluate(self.model, self.ds["val"], self.device, use_depth=self.cfg["model"]["fusion"] != "none",
                          threshold=self.cfg["eval"]["threshold"], num_workers=0)
            score = self.sign * va[self.metric]
            improved = self.best is None or score > self.best
            if improved:
                self.best, self.bad_epochs = score, 0
            else:
                self.bad_epochs += 1
            row = {"epoch": self.epoch + 1, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()},
                   "lr_backbone": self.opt.param_groups[0]["lr"], "lr_head": self.opt.param_groups[1]["lr"],
                   "epoch_seconds": time.time() - e0}
            self.history.append(row)
            for k, v in row.items():
                if k != "epoch":
                    self.tb.add_scalar(k, v, self.epoch + 1)
            print(" | ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)), flush=True)
            self.epoch += 1  # state saved reflects completed epochs
            torch.save(self._state(), self.dir / "last.pt")
            if improved:
                torch.save(self._state(), self.dir / "best.pt")
            self.epoch -= 1
            dump_json(self.history, self.dir / "history.json")
            if self.bad_epochs >= self.t["early_stop_patience"]:
                print(f"early stopping: no {self.metric} improvement for {self.bad_epochs} epochs")
                break
        summary = {"best_val_" + self.metric: self.sign * self.best if self.best is not None else None,
                   "epochs_run": len(self.history), "train_seconds": time.time() - t0, **self.info}
        dump_json(summary, self.dir / "summary.json")
        self.tb.close()
        return summary
