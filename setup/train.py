#!/usr/bin/env python3
"""
Step 1 (PC): Train a YOLO detector using CUDA if available.

- Trains on PC (CUDA preferred, falls back to CPU).
- Produces runs/.../weights/best.pt that you can copy to your Mac for export.

Dataset expectation:
- Roboflow export in "Ultralytics YOLO" format (data.yaml + images/labels folders)

Notes for YOUR downstream inference code:
- Your CoreML pipeline effectively feeds BGR-ordered pixels through PIL (interpreted as RGB).
  To keep your inference code unchanged, we train with bgr=1.0 so the model is robust to that.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def pick_train_device(requested: str) -> str:
    """
    Ultralytics device strings:
      - "0" (CUDA GPU 0), "0,1" for multi-GPU
      - "cpu"
    """
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to Ultralytics data.yaml")
    p.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Base model: yolo26n.pt (recommended), yolo26n.pt, etc. Or a .yaml to train from scratch.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", help='auto | "0" | "0,1" | cpu')
    p.add_argument("--project", type=str, default="runs_detect_coreml")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--patience", type=int, default=50)


    args = p.parse_args()

    data = Path(args.data).expanduser().resolve()
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    device = pick_train_device(args.device)
    print(f"[train.py] device={device} (requested={args.device})")
    print(f"[train.py] model={args.model}")
    print(f"[train.py] data={data}")

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
    )

    # Ultralytics sets this after train()
    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n[train.py] Done. Copy this file to Mac for COREML export:\n  {best_pt}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


