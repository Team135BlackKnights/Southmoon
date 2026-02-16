#!/usr/bin/env python3
"""
Step 2 (macOS): Export a trained YOLO .pt into a CoreML .mlpackage.

- Loads your trained best.pt (copied from PC).
- Exports CoreML package with NMS and (optionally) INT8 quantization.
- Copies the final .mlpackage to a path you choose.

This is meant to produce a package compatible with your inference code that calls:
  self._model.predict({"image": pil})
and reads:
  prediction["coordinates"], prediction["confidence"]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def newest(path: Path, pattern: str) -> Path | None:
    hits = sorted(path.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt copied from PC")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=str, default="detector.mlpackage")

    # Export knobs
    p.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NMS in export when supported (end2end models will force False).",
    )
    p.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FP16 export (recommended).",
    )
    p.add_argument(
        "--int8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="INT8 export (fastest). Requires --data for calibration to be reliable.",
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional: path to data.yaml for INT8 calibration. Strongly recommended if --int8.",
    )

    args = p.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))

    export_kwargs = dict(
        format="coreml",
        imgsz=args.imgsz,
        batch=1,
        nms=args.nms,
        half=args.half,
        int8=args.int8,
        device="cpu",  # export is fine on CPU; CoreML conversion happens locally
    )

    if args.int8:
        if not args.data:
            print("[WARN] --int8 enabled but --data not provided. INT8 may fail; will fall back to FP16 if needed.")
        else:
            data = Path(args.data).expanduser().resolve()
            if not data.exists():
                raise FileNotFoundError(f"data.yaml not found: {data}")
            export_kwargs["data"] = str(data)
 
    print(f"[export_coreml.py] Exporting with: {export_kwargs}")
    try:
        model.export(**export_kwargs)
    except Exception as e:
        if args.int8:
            print("[WARN] INT8 export failed, retrying FP16 only. Error was:", repr(e))
            export_kwargs["int8"] = False
            export_kwargs.pop("data", None)
            export_kwargs.pop("fraction", None)
            model.export(**export_kwargs)
        else:
            raise

    # Ultralytics export puts artifacts near weights/run; search nearby
    search_root = weights.parent.parent if weights.parent.name == "weights" else weights.parent
    mlpackage = newest(search_root, "*.mlpackage")
    if mlpackage is None:
        raise RuntimeError("Could not find exported .mlpackage after export.")

    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    print(f"[export_coreml.py] Copying:\n  from: {mlpackage}\n    to: {out_path}")
    shutil.copytree(mlpackage, out_path)

    # Verify that outputs match ObjectDetector expectations
    try:
        import coremltools as ct  # type: ignore

        mlmodel_path = out_path / "Data" / "com.apple.CoreML" / "model.mlmodel"
        if mlmodel_path.exists():
            spec = ct.models.MLModel(str(mlmodel_path)).get_spec()
            out_names = {o.name for o in spec.description.output}
            missing = {"coordinates", "confidence"} - out_names
            if missing:
                raise RuntimeError(
                    f"CoreML outputs missing {missing}. "
                    "This will not match ObjectDetector.py. "
                    "Use a non-end2end model or export with NMS outputs."
                )
            print(f"[export_coreml.py] Verified outputs: {sorted(out_names)}")
        else:
            print("[WARN] Could not verify outputs: model.mlmodel not found in package.")
    except Exception as e:
        print("[WARN] Output verification failed:", repr(e))

    print("\n[DONE] Final CoreML package:")
    print(f"  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
