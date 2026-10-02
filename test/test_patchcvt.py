from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

################################################################################
# PATH SETUP
################################################################################
# Allow importing the existing PatchCvT implementation from train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import PatchCvT, VALID_EXTENSIONS, yolo_to_xyxy  # noqa: E402


################################################################################
# CONFIGURATION
################################################################################
TEST_DATA_DIR = PROJECT_ROOT / "test" / "test_data"
MODEL_PATH: Optional[Path] = None  # Example: PROJECT_ROOT / "model" / "best.pt"
MEMORY_BANK_PATH: Optional[Path] = None  # Optional external memory tensor path
THRESHOLD = 0.5
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = PROJECT_ROOT / "test" / "results.csv"
OUTPUT_VIS_DIR = PROJECT_ROOT / "test" / "results"
SAVE_VISUALIZATIONS = True
PRESERVATION_AGGREGATION = "mean"  # one of: mean, min, p10


################################################################################
# HELPERS
################################################################################
def parse_annotation_file(label_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse extended YOLO lines:
    class_id x_center y_center width height anomaly_flag

    Returns:
      - parsed annotations
      - malformed line count
    """
    annotations: List[Dict[str, Any]] = []
    malformed = 0

    if not label_path.exists():
        return annotations, malformed

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # pragma: no cover - robust runtime handling
        print(f"[WARN] Cannot read label file {label_path}: {exc}")
        return annotations, 1

    for line_no, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) < 6:
            malformed += 1
            print(
                f"[WARN] Malformed annotation line (<6 fields) in {label_path} "
                f"line {line_no}: {raw}"
            )
            continue

        try:
            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            anomaly = int(float(parts[5]))
        except ValueError:
            malformed += 1
            print(
                f"[WARN] Malformed numeric values in {label_path} "
                f"line {line_no}: {raw}"
            )
            continue

        if anomaly not in (0, 1):
            malformed += 1
            print(
                f"[WARN] anomaly_flag must be 0/1 in {label_path} "
                f"line {line_no}: {raw}"
            )
            continue

        annotations.append(
            {
                "class_id": cls,
                "cx": cx,
                "cy": cy,
                "bw": bw,
                "bh": bh,
                "gt_anomaly": anomaly,
            }
        )

    return annotations, malformed


def collect_images(data_dir: Path) -> List[Path]:
    exts = {ext.lower() for ext in VALID_EXTENSIONS}
    images = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(images)


def find_checkpoint_path() -> Path:
    if MODEL_PATH is not None:
        if MODEL_PATH.exists():
            return MODEL_PATH
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    candidates = [
        PROJECT_ROOT / "model" / "best.pt",
        PROJECT_ROOT / "best.pt",
        PROJECT_ROOT / "model" / "last.pt",
        PROJECT_ROOT / "last.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No checkpoint found. Set MODEL_PATH in test/test_patchcvt.py."
    )


def resolve_memory_bank(
    ckpt: Dict[str, Any],
) -> Tuple[torch.Tensor, str]:
    """
    Resolve memory bank source without rebuilding from test data.
    Priority:
      1) explicit MEMORY_BANK_PATH (if configured)
      2) model_state['memory_bank'] in checkpoint
      3) top-level checkpoint key 'memory_bank'
    """
    if MEMORY_BANK_PATH is not None:
        if not MEMORY_BANK_PATH.exists():
            raise FileNotFoundError(
                f"MEMORY_BANK_PATH does not exist: {MEMORY_BANK_PATH}"
            )
        mem = torch.load(MEMORY_BANK_PATH, map_location="cpu")
        if not isinstance(mem, torch.Tensor):
            raise TypeError("External memory bank must be a torch.Tensor")
        return mem, str(MEMORY_BANK_PATH)

    model_state = ckpt.get("model_state", {})
    if "memory_bank" in model_state and isinstance(model_state["memory_bank"], torch.Tensor):
        return model_state["memory_bank"], "checkpoint:model_state[memory_bank]"

    if "memory_bank" in ckpt and isinstance(ckpt["memory_bank"], torch.Tensor):
        return ckpt["memory_bank"], "checkpoint[memory_bank]"

    raise KeyError(
        "Memory bank not found in checkpoint. Set MEMORY_BANK_PATH to a tensor file."
    )


def aggregate_patch_scores(values: torch.Tensor, mode: str) -> float:
    """
    Aggregate patch-level predictions into ROI-level score.

    Supports tensors shaped as:
      - [1, N, 1] patch-level
      - [1, 1] ROI-level
      - [N] or other flattenable variants
    """
    if values is None:
        return float("nan")

    arr = values.detach().float().cpu().numpy().reshape(-1)
    if arr.size == 0:
        return float("nan")

    if mode == "mean":
        return float(np.mean(arr))
    if mode == "min":
        return float(np.min(arr))
    if mode == "p10":
        return float(np.percentile(arr, 10))

    raise ValueError(f"Unknown aggregation mode: {mode}")


def tensor_shape(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    return type(x).__name__


def print_model_api_discovery(model: torch.nn.Module, device: torch.device) -> None:
    print("=" * 60)
    print("PatchCvT OUTPUT API DISCOVERY")
    print("=" * 60)

    dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(dummy)

    print("Input tensor shape expected by forward: [B, 3, H, W]")
    print(f"Dummy input used: {tuple(dummy.shape)}")

    if not isinstance(out, dict):
        print(f"[WARN] Model output type is {type(out)}, not dict")
        return

    print("Output keys and shapes:")
    for key, value in out.items():
        print(f"  - {key}: shape={tensor_shape(value)}")

    print("Score interpretation check:")
    print("  - train.py applies Sigmoid in RobustnessHead and PreservationHead")
    print("  - preservation output is already in [0,1] (do not apply sigmoid twice)")

    pres = out.get("preservation")
    rob = out.get("robustness")
    gate = out.get("gate")

    if isinstance(pres, torch.Tensor):
        print(f"  - preservation appears ROI-level with shape {tuple(pres.shape)}")
    if isinstance(rob, torch.Tensor):
        print(f"  - robustness appears patch-level with shape {tuple(rob.shape)}")
    if isinstance(gate, torch.Tensor):
        print(f"  - gate appears patch-level with shape {tuple(gate.shape)}")

    print("=" * 60)


def build_preprocess() -> T.Compose:
    # Exact ROI preprocessing used by BaseROIDataset in train.py
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
        ]
    )


def safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((gt == 1) & (pred == 1)))
    tn = int(np.sum((gt == 0) & (pred == 0)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_auc_scores(gt: np.ndarray, preservation_score: np.ndarray) -> Dict[str, Optional[float]]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {"roc_auc": None, "pr_auc": None}

    # AUC requires both classes present
    unique = np.unique(gt)
    if len(unique) < 2:
        return {"roc_auc": None, "pr_auc": None}

    try:
        roc_auc = float(roc_auc_score(gt, preservation_score))
    except Exception:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(gt, preservation_score))
    except Exception:
        pr_auc = None

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def draw_visualizations(
    image: np.ndarray,
    rows_for_image: List[Dict[str, Any]],
    save_path: Path,
) -> None:
    vis = image.copy()

    for row in rows_for_image:
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        gt = int(row["gt_anomaly"])
        pred = int(row["predicted_pass"])
        pres = float(row["preservation_score"])

        gt_pass = int(gt == 0)
        correct = gt_pass == pred
        color = (0, 200, 0) if correct else (0, 0, 255)  # BGR

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        obj_idx = row["object_index"]
        gt_txt = "ANOMALY" if gt == 1 else "NORMAL"
        pred_txt = "PASS" if pred == 1 else "FAIL"

        lines = [
            f"TCD #{obj_idx}",
            f"GT: {gt_txt}",
            f"PatchCvT preservation: {pres:.3f}",
            f"Prediction: {pred_txt}",
        ]

        text_y = max(14, y1 - 6)
        for i, line in enumerate(lines):
            yy = text_y - (len(lines) - 1 - i) * 14
            cv2.putText(
                vis,
                line,
                (x1, max(12, yy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> None:
    print(f"Using device: {DEVICE}")

    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"TEST_DATA_DIR does not exist: {TEST_DATA_DIR}")

    checkpoint_path = find_checkpoint_path()
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint format is unexpected: expected a dict")

    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint missing dict key: model_state")

    memory_bank_cpu, memory_bank_source = resolve_memory_bank(ckpt)
    if memory_bank_cpu.ndim != 2:
        raise ValueError(
            f"Expected memory bank shape [M, C], got {tuple(memory_bank_cpu.shape)}"
        )

    model = PatchCvT(memory_bank=memory_bank_cpu)
    model.load_state_dict(model_state, strict=True)
    model = model.to(DEVICE)
    model.eval()

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Memory bank path/source: {memory_bank_source}")
    print(f"Memory bank shape: {tuple(memory_bank_cpu.shape)}")
    print(f"Model device: {next(model.parameters()).device}")

    print_model_api_discovery(model, DEVICE)

    print("Prediction convention: pass if preservation_score >= THRESHOLD")
    print(f"Preservation threshold: {THRESHOLD:.4f}")

    preprocess = build_preprocess()
    image_paths = collect_images(TEST_DATA_DIR)

    if not image_paths:
        print(f"[WARN] No images found in {TEST_DATA_DIR}")

    stats = {
        "images_total": len(image_paths),
        "images_skipped": 0,
        "malformed_labels": 0,
        "invalid_boxes": 0,
        "inference_failures": 0,
        "skipped_annotations": 0,
    }

    rows: List[Dict[str, Any]] = []

    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        annotations, malformed = parse_annotation_file(label_path)
        stats["malformed_labels"] += malformed

        if len(annotations) == 0:
            stats["images_skipped"] += 1
            if not label_path.exists():
                print(f"[WARN] Missing label file for {image_path.name}: {label_path.name}")
            else:
                print(f"[WARN] No valid annotations in {label_path.name}")
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            stats["images_skipped"] += 1
            print(f"[WARN] Cannot read image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        image_rows: List[Dict[str, Any]] = []

        for obj_idx, ann in enumerate(annotations):
            box = [
                ann["class_id"],
                ann["cx"],
                ann["cy"],
                ann["bw"],
                ann["bh"],
            ]

            x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)

            if x2 <= x1 or y2 <= y1:
                stats["invalid_boxes"] += 1
                stats["skipped_annotations"] += 1
                print(
                    f"[WARN] Invalid bbox in {label_path.name} object {obj_idx}: "
                    f"({x1},{y1},{x2},{y2})"
                )
                continue

            roi = image_rgb[y1:y2, x1:x2]
            if roi.size == 0:
                stats["invalid_boxes"] += 1
                stats["skipped_annotations"] += 1
                print(
                    f"[WARN] Empty ROI in {label_path.name} object {obj_idx}: "
                    f"({x1},{y1},{x2},{y2})"
                )
                continue

            try:
                roi_pil = Image.fromarray(roi)
                roi_tensor = preprocess(roi_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = model(roi_tensor)

                if not isinstance(out, dict):
                    raise TypeError(f"Model output must be dict, got {type(out)}")

                preservation_tensor = out.get("preservation")
                robustness_tensor = out.get("robustness")
                residual_tensor = out.get("residual")
                gate_tensor = out.get("gate")

                if not isinstance(preservation_tensor, torch.Tensor):
                    raise KeyError("Model output missing tensor key: 'preservation'")

                preservation_score = aggregate_patch_scores(
                    preservation_tensor,
                    PRESERVATION_AGGREGATION,
                )

                # Robustness/gate/residual are patch-level in current train.py
                robustness_score = (
                    aggregate_patch_scores(robustness_tensor, "mean")
                    if isinstance(robustness_tensor, torch.Tensor)
                    else float("nan")
                )
                residual_score = (
                    aggregate_patch_scores(residual_tensor, "mean")
                    if isinstance(residual_tensor, torch.Tensor)
                    else float("nan")
                )
                gate_score = (
                    aggregate_patch_scores(gate_tensor, "mean")
                    if isinstance(gate_tensor, torch.Tensor)
                    else float("nan")
                )

                predicted_pass = int(preservation_score >= THRESHOLD)

                row = {
                    "image": image_path.name,
                    "object_index": obj_idx,
                    "class_id": ann["class_id"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "gt_anomaly": ann["gt_anomaly"],
                    "robustness_score": robustness_score,
                    "preservation_score": preservation_score,
                    "predicted_pass": predicted_pass,
                    "residual_score": residual_score,
                    "gate_score": gate_score,
                    "preservation_aggregation": PRESERVATION_AGGREGATION,
                }

                rows.append(row)
                image_rows.append(row)

            except Exception as exc:  # pragma: no cover - robust runtime handling
                stats["inference_failures"] += 1
                stats["skipped_annotations"] += 1
                print(
                    f"[WARN] Inference failure for {image_path.name} object {obj_idx}: {exc}"
                )

        if SAVE_VISUALIZATIONS and image_rows:
            vis_name = f"{image_path.stem}_annotated{image_path.suffix}"
            vis_path = OUTPUT_VIS_DIR / vis_name
            try:
                draw_visualizations(image_rgb, image_rows, vis_path)
            except Exception as exc:  # pragma: no cover - robust runtime handling
                print(f"[WARN] Failed visualization for {image_path.name}: {exc}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "object_index",
        "class_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "gt_anomaly",
        "robustness_score",
        "preservation_score",
        "predicted_pass",
        "residual_score",
        "gate_score",
        "preservation_aggregation",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) == 0:
        print("=" * 60)
        print("PatchCvT BENCHMARK")
        print("=" * 60)
        print("No valid ROI predictions were produced.")
        print(f"Images scanned: {stats['images_total']}")
        print(f"Skipped images: {stats['images_skipped']}")
        print(f"Malformed labels: {stats['malformed_labels']}")
        print(f"Invalid bounding boxes: {stats['invalid_boxes']}")
        print(f"Inference failures: {stats['inference_failures']}")
        print(f"Results CSV: {OUTPUT_CSV}")
        return

    gt_anomaly = np.array([int(r["gt_anomaly"]) for r in rows], dtype=np.int64)
    gt = (gt_anomaly == 0).astype(np.int64)
    pred = np.array([int(r["predicted_pass"]) for r in rows], dtype=np.int64)
    preservation_score = np.array([float(r["preservation_score"]) for r in rows], dtype=np.float64)

    binary = compute_binary_metrics(gt, pred)
    auc = compute_auc_scores(gt, preservation_score)

    normal_mask = gt_anomaly == 0
    anomaly_mask = gt_anomaly == 1

    mean_pres_normal = float(np.mean([rows[i]["preservation_score"] for i in np.where(normal_mask)[0]])) if np.any(normal_mask) else float("nan")
    mean_pres_anom = float(np.mean([rows[i]["preservation_score"] for i in np.where(anomaly_mask)[0]])) if np.any(anomaly_mask) else float("nan")

    mean_rob_normal = float(np.mean([rows[i]["robustness_score"] for i in np.where(normal_mask)[0]])) if np.any(normal_mask) else float("nan")
    mean_rob_anom = float(np.mean([rows[i]["robustness_score"] for i in np.where(anomaly_mask)[0]])) if np.any(anomaly_mask) else float("nan")

    print("=" * 60)
    print("PatchCvT BENCHMARK")
    print("=" * 60)
    print(f"Images evaluated:       {stats['images_total'] - stats['images_skipped']}")
    print(f"ROIs evaluated:         {len(rows)}")
    print(f"Normal ROIs:            {int(np.sum(gt_anomaly == 0))}")
    print(f"Anomalous ROIs:         {int(np.sum(gt_anomaly == 1))}")
    print()

    print(f"Preservation threshold: {THRESHOLD:.2f}")

    print()
    print(f"Accuracy:               {binary['accuracy']:.6f}")
    print(f"Precision:              {binary['precision']:.6f}")
    print(f"Recall:                 {binary['recall']:.6f}")
    print(f"F1:                     {binary['f1']:.6f}")

    roc_txt = f"{auc['roc_auc']:.6f}" if auc["roc_auc"] is not None else "N/A"
    pr_txt = f"{auc['pr_auc']:.6f}" if auc["pr_auc"] is not None else "N/A"
    print(f"ROC-AUC:                {roc_txt}")
    print(f"PR-AUC:                 {pr_txt}")
    print()

    print("Mean preservation:")
    print(f"    normal:             {mean_pres_normal:.6f}")
    print(f"    anomalous:          {mean_pres_anom:.6f}")
    print()

    print("Mean robustness:")
    print(f"    normal:             {mean_rob_normal:.6f}")
    print(f"    anomalous:          {mean_rob_anom:.6f}")
    print()

    print("Confusion matrix counts:")
    print(f"    TN:                 {binary['tn']}")
    print(f"    FP:                 {binary['fp']}")
    print(f"    FN:                 {binary['fn']}")
    print(f"    TP:                 {binary['tp']}")
    print()

    print("Robustness report:")
    print(f"    Successfully evaluated ROIs: {len(rows)}")
    print(f"    Skipped images:              {stats['images_skipped']}")
    print(f"    Skipped annotations:         {stats['skipped_annotations']}")
    print(f"    Malformed labels:            {stats['malformed_labels']}")
    print(f"    Invalid bounding boxes:      {stats['invalid_boxes']}")
    print(f"    Inference failures:          {stats['inference_failures']}")
    print()

    print(f"Results CSV: {OUTPUT_CSV}")
    if SAVE_VISUALIZATIONS:
        print(f"Visualization directory: {OUTPUT_VIS_DIR}")


if __name__ == "__main__":
    main()
