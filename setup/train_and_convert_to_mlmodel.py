#!/usr/bin/env python3
"""
Train + export a CoreML .mlpackage object detector for your existing CoreMLObjectDetector.

Key goals:
- Fast on Apple Silicon: use a nano YOLO + CoreML export with NMS + quantization.
- Keep your inference code unchanged:
  - Your code effectively feeds BGR-ordered data through PIL (interpreted as RGB).
  - We train with bgr=1.0 augmentation so the network learns to tolerate/expect that ordering.

Dataset expectation:
- Ultralytics YOLO dataset layout with a data.yaml (from Roboflow "YOLO" export).
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path


def pick_device(requested: str) -> str:
    """
    Ultralytics accepts:
      - "0" or "0,1" for CUDA GPUs
      - "mps" for Apple Silicon
      - "cpu"
    """
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def newest(path: Path, pattern: str) -> Path | None:
    hits = sorted(path.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def print_coreml_io(model_path: Path) -> None:
    """
    Prints model inputs/outputs. On non-macOS, CoreML prediction won't run,
    but we can still inspect the spec names.
    """
    try:
        import coremltools as ct  # type: ignore

        m = ct.models.MLModel(str(model_path))
        spec = m.get_spec()
        ins = [f"{i.name}:{i.type.WhichOneof('Type')}" for i in spec.description.input]
        outs = [f"{o.name}:{o.type.WhichOneof('Type')}" for o in spec.description.output]
        print("\n[CoreML] train.py export inspection")
        print("  Inputs :", ins)
        print("  Outputs:", outs)

        # Try a dummy predict only on macOS (CoreML runtime requirement)
        if platform.system().lower() == "darwin":
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore

            img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
            try:
                out = m.predict({"image": img})
                print("  predict() keys:", list(out.keys()))
            except Exception as e:
                print("  predict() failed (may require additional inputs):", repr(e))
    except Exception as e:
        print("[CoreML] Could not inspect model IO:", repr(e))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to Ultralytics data.yaml")
    p.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Base model checkpoint (e.g., yolo26n.pt, yolo11n.pt, or your own .pt)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=str, default="runs_detect_coreml")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--patience", type=int, default=50)

    # Important for *your* unchanged inference pipeline:
    p.add_argument(
        "--bgr",
        type=float,
        default=1.0,
        help="Ultralytics augmentation: flip RGB<->BGR with probability. "
             "Set to 1.0 to match your current OpenCV->PIL channel ordering.",
    )

    # Export knobs
    p.add_argument("--export", action="store_true", default=True)
    p.add_argument("--nms", action="store_true", default=True)
    p.add_argument("--half", action="store_true", default=True, help="FP16 export")
    p.add_argument(
        "--int8",
        action="store_true",
        default=True,
        help="INT8 export (fastest). Requires representative data.yaml for calibration.",
    )
    p.add_argument(
        "--calib-fraction",
        type=float,
        default=0.2,
        help="Fraction of dataset used for INT8 calibration (0-1).",
    )
    p.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Force end2end mode (NMS-free). Leave OFF for compatibility with nms=True pipeline.",
    )

    p.add_argument(
        "--out",
        type=str,
        default="detector.mlpackage",
        help="Where to copy the final .mlpackage",
    )

    args = p.parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.exists():
        print(f"[ERROR] data.yaml not found: {data}")
        return 2

    device = pick_device(args.device)
    print(f"[train.py] Using device: {device}")
    print(f"[train.py] Base model: {args.model}")
    print(f"[train.py] Dataset: {data}")

    # Train
    from ultralytics import YOLO  # type: ignore

    model = YOLO(args.model)
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        bgr=args.bgr,
    )

    # Find best.pt
    save_dir = Path(model.trainer.save_dir)  # set by ultralytics
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] best.pt not found at: {best_pt}")
        return 3

    print(f"[train.py] Training complete. best.pt = {best_pt}")

    if not args.export:
        return 0

    # Export
    export_model = YOLO(str(best_pt))

    export_kwargs = dict(
        format="coreml",
        imgsz=args.imgsz,
        device=device,
        batch=1,
        nms=args.nms,
        half=args.half,
        int8=args.int8,
        end2end=args.end2end,
    )

    # Per Ultralytics docs, INT8 calibration uses a representative dataset via `data=...`. :contentReference[oaicite:2]{index=2}
    if args.int8:
        export_kwargs["data"] = str(data)
        export_kwargs["fraction"] = float(args.calib_fraction)

    print(f"[train.py] Exporting CoreML with: {export_kwargs}")
    try:
        exported = export_model.export(**export_kwargs)
    except Exception as e:
        # If INT8 fails (common if calibration deps/environment mismatch), fall back to FP16.
        print("[WARN] CoreML export failed:", repr(e))
        if args.int8:
            print("[WARN] Retrying export with INT8 disabled (FP16 only).")
            export_kwargs["int8"] = False
            export_kwargs.pop("data", None)
            export_kwargs.pop("fraction", None)
            exported = export_model.export(**export_kwargs)
        else:
            raise

    # exported may be a string/path; locate the newest .mlpackage in the run directory.
    mlpackage = newest(save_dir, "*.mlpackage")
    if mlpackage is None:
        # sometimes export writes next to weights
        mlpackage = newest(best_pt.parent, "*.mlpackage")

    if mlpackage is None:
        print("[ERROR] Could not find exported .mlpackage")
        return 4

    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    print(f"[train.py] Copying exported package:\n  from: {mlpackage}\n    to: {out_path}")
    shutil.copytree(mlpackage, out_path)

    print_coreml_io(out_path)

    print("\n[DONE] Final model package:")
    print(f"  {out_path}")
    print("Use this path as `model_path` in your existing CoreMLObjectDetector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
