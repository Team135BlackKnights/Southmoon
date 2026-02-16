#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Ultralytics YOLO
from ultralytics import YOLO


def draw_detections(
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    names: dict,
    conf_thresh: float,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, conf, cls):
        c = float(c)
        if c < conf_thresh:
            continue

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        class_id = int(k)
        label = names.get(class_id, str(class_id))
        text = f"{label} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return out


def run_on_frame(
    model: YOLO,
    frame_bgr: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    verbose: bool,
):
    # Ultralytics expects BGR ndarray fine; it handles preprocessing internally.
    # We set stream=False to get a single Results object list.
    results = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=verbose,
        device=None,  # let ultralytics decide
    )
    return results


def extract_first_results(results):
    """
    results is a list[ultralytics.engine.results.Results]
    """
    if not results:
        return None
    r0 = results[0]
    return r0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained .pt (best.pt)")
    ap.add_argument("--source", default="0", help="Webcam index (0) or path to image/video")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--show", action="store_true", help="Show OpenCV window")
    ap.add_argument("--save", default="", help="Optional output video path (mp4)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print-raw", action="store_true", help="Print raw tensor stats each frame")
    args = ap.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        print(f"[ERROR] weights not found: {weights}")
        return 2

    print(f"[INFO] Loading model: {weights}")
    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    if names is None:
        # Ultralytics sometimes stores in model.model.names
        names = getattr(getattr(model, "model", None), "names", {})
    if not isinstance(names, dict):
        # can be list
        names = {i: n for i, n in enumerate(list(names))}

    print(f"[INFO] Classes: {len(names)} -> {names}")

    # Decide source
    src = args.source
    is_cam = False
    cap = None

    if src.isdigit():
        is_cam = True
        cam_index = int(src)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam index {cam_index}")
            return 3
    else:
        src_path = Path(src).expanduser().resolve()
        if not src_path.exists():
            print(f"[ERROR] source not found: {src_path}")
            return 4

        # Image?
        if src_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
            frame = cv2.imread(str(src_path))
            if frame is None:
                print("[ERROR] cv2.imread failed")
                return 5

            t0 = time.time()
            results = run_on_frame(model, frame, args.imgsz, args.conf, args.iou, args.max_det, args.verbose)
            dt = (time.time() - t0) * 1000.0
            r0 = extract_first_results(results)
            if r0 is None:
                print("[ERROR] No results object returned")
                return 6

            boxes = r0.boxes
            n = 0 if boxes is None else len(boxes)
            print(f"[INFO] Inference time: {dt:.1f} ms, detections: {n}")

            if boxes is not None and n > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                if args.print_raw:
                    print("[RAW] xyxy:", xyxy[:3])
                    print("[RAW] conf:", confs[:10])
                    print("[RAW] cls :", clss[:10])

                vis = draw_detections(frame, xyxy, confs, clss, names, args.conf)
            else:
                vis = frame

            if args.show:
                cv2.imshow("PT Test", vis)
                cv2.waitKey(0)
            if args.save:
                out_path = Path(args.save).expanduser().resolve()
                cv2.imwrite(str(out_path), vis)
                print(f"[INFO] Saved annotated image to: {out_path}")
            return 0

        # Otherwise treat as video file
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {src_path}")
            return 7

    # Video/webcam loop
    writer = None
    out_path = Path(args.save).expanduser().resolve() if args.save else None
    fps_smooth = None
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t0 = time.time()
        results = run_on_frame(model, frame, args.imgsz, args.conf, args.iou, args.max_det, args.verbose)
        r0 = extract_first_results(results)
        infer_ms = (time.time() - t0) * 1000.0

        det_count = 0
        vis = frame

        if r0 is not None and getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
            boxes = r0.boxes
            det_count = len(boxes)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            if args.print_raw:
                print(f"[RAW] det={det_count} infer_ms={infer_ms:.1f} conf_max={float(np.max(confs)):.3f}")

            vis = draw_detections(frame, xyxy, confs, clss, names, args.conf)
        else:
            if args.print_raw:
                print(f"[RAW] det=0 infer_ms={infer_ms:.1f}")

        # FPS estimate
        now = time.time()
        frame_dt = now - last
        last = now
        inst_fps = 1.0 / max(frame_dt, 1e-6)
        fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)

        cv2.putText(
            vis,
            f"det={det_count} infer={infer_ms:.1f}ms fps~{fps_smooth:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Setup writer once we know frame size
        if out_path and writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))
            if not writer.isOpened():
                print("[WARN] Could not open VideoWriter, disabling save.")
                writer = None

        if writer is not None:
            writer.write(vis)

        if args.show:
            cv2.imshow("PT Test", vis)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break

    # Cleanup
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    