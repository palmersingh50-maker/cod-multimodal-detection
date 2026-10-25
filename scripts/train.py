import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cod.config import load_config
from cod.training.trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description="Train a COD model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=5 model.fusion=gated")
    a = ap.parse_args()
    Trainer(load_config(a.config, a.overrides), a.device, a.resume).fit()


if __name__ == "__main__":
    main()
