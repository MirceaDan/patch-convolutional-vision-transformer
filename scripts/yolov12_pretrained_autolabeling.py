from pathlib import Path
import os
import shutil
import sys
import time


from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing your original images.
# The script searches recursively through all subfolders.
INPUT_ROOT = Path(r"J:\TCDNet\ExtractedFrames\260503_183425_497_TCD_Zimbor")

# Folder where the generated dataset will be created.
OUTPUT_ROOT = Path(r"E:\Projects\patch-convolutional-vision-transformer\database\260503_183425_497_TCD_Zimbor")

# YOLO confidence threshold.
#
# The pretrained model has relatively low recall (~0.419),
# so we deliberately start lower than the usual 0.25.
#
# You can later experiment with:
#   0.10
#   0.15
#   0.20
#   0.25
CONF_THRESHOLD = 0.01

# YOLO inference image size.
# The pretrained model was trained at 640x640.
IMAGE_SIZE = 640

# GPU device.
# 
# "0"  -> first NVIDIA GPU
# "1"  -> second NVIDIA GPU
# "cpu" -> CPU
DEVICE = "cpu"

# Batch size.
#
# YOLOv12x is a large model, so start conservatively.
BATCH_SIZE = 1

# Supported image extensions.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# ============================================================
# CPU RESOURCE LIMITS
# ============================================================

# Maximum number of CPU threads PyTorch may use for operations.
# Start conservatively. Increase only if the machine has plenty
# of CPU capacity available.
TORCH_THREADS = 2

# Number of threads used for PyTorch inter-op parallelism.
TORCH_INTEROP_THREADS = 1

# Limit common native math libraries as well.
os.environ["OMP_NUM_THREADS"] = str(TORCH_THREADS)
os.environ["MKL_NUM_THREADS"] = str(TORCH_THREADS)

import torch

torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(TORCH_INTEROP_THREADS)

# ============================================================
# MODEL
# ============================================================

HF_REPO = "maco018/YOLOv12_traffic-delineator"
MODEL_FILENAME = "YOLOv12_traffic-delineator.pt"


# ============================================================
# HELPERS
# ============================================================

def find_images(root: Path):
    """Recursively find all supported images."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_path_for(input_path: Path) -> Path:
    """
    Convert an input image path to its corresponding output path
    while preserving the directory structure.
    """

    relative_path = input_path.relative_to(INPUT_ROOT)

    return OUTPUT_ROOT / relative_path


def write_yolo_labels(txt_path: Path, boxes):
    """
    Write YOLO annotations.

    Format:
        class_id x_center y_center width height

    All coordinates are normalized to [0, 1].
    """

    with txt_path.open("w", encoding="utf-8") as f:

        for box in boxes:

            # xywhn = normalized:
            # x_center, y_center, width, height
            x_center, y_center, width, height = box.tolist()

            # IMPORTANT:
            # We intentionally force the class to 0.
            f.write(
                f"0 "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLOv12 Traffic Delineator Auto-Labeler")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_ROOT.exists():
        print(f"\nERROR: Input folder does not exist:")
        print(INPUT_ROOT)
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    print("\nSearching for images...")
    images = find_images(INPUT_ROOT)

    print(f"Found {len(images):,} images.")

    if not images:
        print("No supported images found.")
        sys.exit(1)
    
    # ============================================================
    # MODEL
    # ============================================================
    print("\nDloading YOLOv12 traffic-delineator model...")
     
    model = YOLO(f'E:\Projects\patch-convolutional-vision-transformer\scripts\YOLOv12_traffic-delineator.pt')

    print("\nModel loaded.")
    print(f"Classes: {model.names}")

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    total_images = len(images)
    total_detections = 0
    images_with_detections = 0
    images_without_detections = 0

    print("\nStarting inference...")
    print("-" * 70)

    # Ultralytics accepts a list of image paths and streams results.
    # This avoids loading the whole dataset into RAM.
    for index, input_path in enumerate(images, start=1):

        start_time = time.perf_counter()

        results = model.predict(
            source=str(input_path),
            conf=CONF_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        elapsed = time.perf_counter() - start_time

        if result.boxes is not None and len(result.boxes) > 0:
            num_detections = len(result.boxes)
        else:
            num_detections = 0

        avg_time = elapsed / index
        remaining = total_images - index
        eta_seconds = remaining * avg_time

        print(
            f"[{index:>4}/{total_images}] "
            f"{input_path.name} | "
            f"detections={num_detections} | "
            f"avg={avg_time:.2f}s/img | "
            f"ETA={eta_seconds / 60:.1f} min",
            flush=True
        )

        # ----------------------------------------------------
        # Determine output location
        # ----------------------------------------------------

        output_image = output_path_for(input_path)
        output_txt = output_image.with_suffix(".txt")

        output_image.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Copy original image
        # ----------------------------------------------------

        shutil.copy2(
            input_path,
            output_image,
        )

        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes.xywhn.cpu()

            num_detections = len(boxes)

            total_detections += num_detections
            images_with_detections += 1

            write_yolo_labels(
                output_txt,
                boxes,
            )

        else:

            # IMPORTANT:
            # Create an EMPTY txt file.
            #
            # This means:
            # "This image was inspected by the auto-labeling
            # pipeline but YOLO found nothing."
            #
            # You can then manually annotate it in LabelImg.
            output_txt.touch()

            images_without_detections += 1

            num_detections = 0

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 100 == 0 or index == total_images:

            percent = index / total_images * 100

            print(
                f"[{index:>7,}/{total_images:,}] "
                f"{percent:6.2f}% | "
                f"detections: {num_detections:2d} | "
                f"{input_path.name}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Images processed:          {total_images:,}")
    print(f"Images with detections:    {images_with_detections:,}")
    print(f"Images without detections: {images_without_detections:,}")
    print(f"Total detections:          {total_detections:,}")

    if total_images > 0:

        detection_rate = (
            images_with_detections / total_images * 100
        )

        average_detections = (
            total_detections / total_images
        )

        print(f"Detection rate:            {detection_rate:.2f}%")
        print(f"Avg detections/image:      {average_detections:.3f}")

    print(f"\nOutput dataset:")
    print(OUTPUT_ROOT)

    print("\nYou can now open the output dataset in LabelImg")
    print("and visually inspect/correct the generated annotations.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()