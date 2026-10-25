"""Export a checkpoint to ONNX (fixed input size) and verify numerical parity with PyTorch via onnxruntime."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import cod.models.depth_encoder as depth_encoder
from cod.inference.predictor import load_checkpoint


class Wrap(torch.nn.Module):
    def __init__(self, m, with_depth):
        super().__init__()
        self.m, self.with_depth = m, with_depth

    def forward(self, image, depth=None):
        if self.with_depth:
            valid = torch.ones(image.shape[0], device=image.device)
            return torch.sigmoid(self.m(image, depth, valid)["logits"])
        return torch.sigmoid(self.m(image)["logits"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()
    model, cfg = load_checkpoint(a.checkpoint, torch.device("cpu"))
    s = cfg["data"]["image_size"]
    with_depth = cfg["model"]["fusion"] != "none"
    w = Wrap(model, with_depth).eval()
    args = (torch.randn(1, 3, s, s),) + ((torch.rand(1, 1, s, s),) if with_depth else ())
    with torch.no_grad():
        ref_aa = w(*args).numpy()
    depth_encoder.ONNX_COMPAT = True  # non-antialiased token-grid resize (exportable)
    names = ["image"] + (["depth"] if with_depth else [])
    torch.onnx.export(w, args, a.out, input_names=names, output_names=["probability"], opset_version=a.opset, dynamo=False)
    print("exported", a.out, f"{Path(a.out).stat().st_size / 2**20:.1f} MB")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipping parity check (pip install onnxruntime)")
        return
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = w(*args).numpy()
    got = sess.run(None, {n: t.numpy() for n, t in zip(names, args)})[0]
    print("max |onnx - torch(same resize)| =", float(np.abs(ref - got).max()))
    print("max |onnx - torch(antialiased resize, as trained)| =", float(np.abs(ref_aa - got).max()))


if __name__ == "__main__":
    main()
