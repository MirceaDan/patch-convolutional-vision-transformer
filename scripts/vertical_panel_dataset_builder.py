from pathlib import Path
import random
import shutil
import sys


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# SOURCE DATASET
# ------------------------------------------------------------
#
# Expected structure:
#
# SOURCE_ROOT/
#     images/
#         image001.jpg
#         image002.jpg
#         ...
#
#     labels/
#         image001.txt
#         image002.txt
#         ...
#
SOURCE_ROOT = Path(
    r"J:\TCDNet\Bbox_Complete_Dataset_Thesis.v1-vertical-panels.yolov7pytorch"
)


# ------------------------------------------------------------
# OUTPUT DATASET
# ------------------------------------------------------------
#
# The script creates:
#
# OUTPUT_ROOT/
#     data.yaml
#
#     train/
#         images/
#         labels/
#
#     val/
#         images/
#         labels/
#
#     test/
#         images/
#         labels/
#
OUTPUT_ROOT = Path(
    r"J:\TCDNet\Roboflow_TCD"
)


# ------------------------------------------------------------
# YOLO CLASS ID IN ORIGINAL DATASET
# ------------------------------------------------------------

VERTICAL_PANEL_CLASS_ID = 13


# ------------------------------------------------------------
# OUR NEW CLASS ID
# ------------------------------------------------------------

TARGET_CLASS_ID = 0


# ------------------------------------------------------------
# DATASET SPLIT
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
TEST_RATIO = 0.20
VAL_RATIO = 0.10

RANDOM_SEED = 42


# ------------------------------------------------------------
# IMAGE EXTENSIONS
# ------------------------------------------------------------

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
# HELPERS
# ============================================================

def find_images(images_root: Path):
    """
    Recursively find all supported images.
    """

    return sorted(
        p
        for p in images_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_label_path(
    image_path: Path,
    images_root: Path,
    labels_root: Path,
):
    """
    Map:

        images/foo/bar/image.jpg

    to:

        labels/foo/bar/image.txt

    preserving the relative directory structure.
    """

    relative_path = image_path.relative_to(images_root)

    return (
        labels_root
        / relative_path.with_suffix(".txt")
    )


def read_vertical_panel_annotations(label_path: Path):
    """
    Read a YOLO annotation file and return ONLY
    Vertical Panel annotations.

    Original:
        13 x y w h

    Returned:
        0 x y w h

    All other classes are ignored.
    """

    vertical_panel_annotations = []

    if not label_path.exists():
        return vertical_panel_annotations

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                print(
                    f"WARNING: malformed annotation "
                    f"in {label_path} "
                    f"(line {line_number})"
                )

                continue

            try:

                class_id = int(parts[0])

            except ValueError:

                print(
                    f"WARNING: invalid class ID "
                    f"in {label_path} "
                    f"(line {line_number})"
                )

                continue

            # ------------------------------------------------
            # Keep ONLY Vertical Panel
            # ------------------------------------------------

            if class_id == VERTICAL_PANEL_CLASS_ID:

                # Force class ID to our new class 0.
                vertical_panel_annotations.append(
                    f"{TARGET_CLASS_ID} "
                    f"{parts[1]} "
                    f"{parts[2]} "
                    f"{parts[3]} "
                    f"{parts[4]}"
                )

    return vertical_panel_annotations


def copy_dataset_item(
    image_path: Path,
    annotations,
    split_name: str,
    images_root: Path,
):
    """
    Copy image and write filtered annotation.
    """

    relative_path = image_path.relative_to(images_root)

    destination_image = (
        OUTPUT_ROOT
        / split_name
        / "images"
        / relative_path
    )

    destination_label = (
        OUTPUT_ROOT
        / split_name
        / "labels"
        / relative_path.with_suffix(".txt")
    )

    # --------------------------------------------------------
    # Create parent directories
    # --------------------------------------------------------

    destination_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_label.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        destination_image,
    )

    # --------------------------------------------------------
    # Write ONLY Vertical Panel annotations
    # --------------------------------------------------------

    with destination_label.open(
        "w",
        encoding="utf-8",
    ) as f:

        for annotation in annotations:

            f.write(
                annotation + "\n"
            )


def write_data_yaml():
    """
    Create YOLO dataset configuration.
    """

    yaml_path = OUTPUT_ROOT / "data.yaml"

    # Use forward slashes because they are safer in YAML.
    root_path = OUTPUT_ROOT.as_posix()

    yaml_content = f"""path: {root_path}

train: train/images
val: val/images
test: test/images

names:
  0: vertical_panel
"""

    with yaml_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(yaml_content)

    return yaml_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Vertical Panel Dataset Builder")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate split configuration
    # --------------------------------------------------------

    split_sum = (
        TRAIN_RATIO
        + TEST_RATIO
        + VAL_RATIO
    )

    if abs(split_sum - 1.0) > 1e-6:

        print(
            "\nERROR: TRAIN_RATIO + TEST_RATIO + VAL_RATIO "
            "must equal 1.0"
        )

        print(
            f"Current sum: {split_sum}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Validate source directories
    # --------------------------------------------------------

    images_root = SOURCE_ROOT / "images"
    labels_root = SOURCE_ROOT / "labels"

    if not SOURCE_ROOT.exists():

        print(
            "\nERROR: SOURCE_ROOT does not exist:"
        )

        print(SOURCE_ROOT)

        sys.exit(1)

    if not images_root.exists():

        print(
            "\nERROR: images directory does not exist:"
        )

        print(images_root)

        sys.exit(1)

    if not labels_root.exists():

        print(
            "\nERROR: labels directory does not exist:"
        )

        print(labels_root)

        sys.exit(1)

    # --------------------------------------------------------
    # Find all images
    # --------------------------------------------------------

    print("\nSearching for images...")

    images = find_images(images_root)

    print(
        f"Found {len(images):,} images."
    )

    if not images:

        print(
            "\nERROR: No images found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Find images containing Vertical Panel
    # --------------------------------------------------------

    print(
        "\nSearching for Vertical Panel annotations..."
    )

    valid_samples = []

    missing_labels = 0
    malformed_or_empty = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        label_path = get_label_path(
            image_path,
            images_root,
            labels_root,
        )

        if not label_path.exists():

            missing_labels += 1

            continue

        annotations = read_vertical_panel_annotations(
            label_path
        )

        if not annotations:

            malformed_or_empty += 1

            continue

        # Store both image path and filtered annotations.
        valid_samples.append(
            (
                image_path,
                annotations,
            )
        )

        # Progress every 500 images.
        if index % 500 == 0:

            print(
                f"Scanned {index:,}/{len(images):,}..."
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FILTERING RESULTS")
    print("=" * 70)

    print(
        f"Total images:              {len(images):,}"
    )

    print(
        f"Images with Vertical Panel: "
        f"{len(valid_samples):,}"
    )

    print(
        f"Images without Vertical Panel: "
        f"{malformed_or_empty:,}"
    )

    print(
        f"Images without label file: "
        f"{missing_labels:,}"
    )

    if not valid_samples:

        print(
            "\nERROR: No Vertical Panel images found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Shuffle deterministically
    # --------------------------------------------------------

    print(
        "\nShuffling dataset..."
    )

    random.seed(RANDOM_SEED)

    random.shuffle(valid_samples)

    # --------------------------------------------------------
    # Calculate split
    # --------------------------------------------------------

    total_samples = len(valid_samples)

    train_count = int(
        total_samples * TRAIN_RATIO
    )

    test_count = int(
        total_samples * TEST_RATIO
    )

    # Whatever remains goes to validation.
    #
    # This guarantees:
    #
    # train + test + val == total
    #
    val_count = (
        total_samples
        - train_count
        - test_count
    )

    # --------------------------------------------------------
    # Create split lists
    # --------------------------------------------------------

    train_samples = valid_samples[
        :train_count
    ]

    test_samples = valid_samples[
        train_count:
        train_count + test_count
    ]

    val_samples = valid_samples[
        train_count + test_count:
    ]

    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total Vertical Panel images: {total_samples:,}"
    )

    print(
        f"Train:                       "
        f"{len(train_samples):,} "
        f"({len(train_samples) / total_samples * 100:.1f}%)"
    )

    print(
        f"Test:                        "
        f"{len(test_samples):,} "
        f"({len(test_samples) / total_samples * 100:.1f}%)"
    )

    print(
        f"Validation:                  "
        f"{len(val_samples):,} "
        f"({len(val_samples) / total_samples * 100:.1f}%)"
    )

    # --------------------------------------------------------
    # Remove old output dataset
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\nWARNING:"
        )

        print(
            "Output dataset already exists:"
        )

        print(
            OUTPUT_ROOT
        )

        print(
            "\nIt will be completely replaced."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BUILDING TRAIN SET")
    print("=" * 70)

    for index, (
        image_path,
        annotations,
    ) in enumerate(
        train_samples,
        start=1,
    ):

        copy_dataset_item(
            image_path=image_path,
            annotations=annotations,
            split_name="train",
            images_root=images_root,
        )

        if (
            index % 100 == 0
            or index == len(train_samples)
        ):

            print(
                f"[TRAIN] "
                f"{index:,}/{len(train_samples):,}",
                flush=True,
            )

    # --------------------------------------------------------
    # Copy TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BUILDING TEST SET")
    print("=" * 70)

    for index, (
        image_path,
        annotations,
    ) in enumerate(
        test_samples,
        start=1,
    ):

        copy_dataset_item(
            image_path=image_path,
            annotations=annotations,
            split_name="test",
            images_root=images_root,
        )

        if (
            index % 100 == 0
            or index == len(test_samples)
        ):

            print(
                f"[TEST]  "
                f"{index:,}/{len(test_samples):,}",
                flush=True,
            )

    # --------------------------------------------------------
    # Copy VALIDATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BUILDING VALIDATION SET")
    print("=" * 70)

    for index, (
        image_path,
        annotations,
    ) in enumerate(
        val_samples,
        start=1,
    ):

        copy_dataset_item(
            image_path=image_path,
            annotations=annotations,
            split_name="val",
            images_root=images_root,
        )

        if (
            index % 100 == 0
            or index == len(val_samples)
        ):

            print(
                f"[VAL]    "
                f"{index:,}/{len(val_samples):,}",
                flush=True,
            )

    # --------------------------------------------------------
    # Create YAML
    # --------------------------------------------------------

    yaml_path = write_data_yaml()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nOutput dataset:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nStructure:"
    )

    print(
        "  train/images/"
    )

    print(
        "  train/labels/"
    )

    print(
        "  val/images/"
    )

    print(
        "  val/labels/"
    )

    print(
        "  test/images/"
    )

    print(
        "  test/labels/"
    )

    print(
        f"\nYAML:"
    )

    print(
        yaml_path
    )

    print(
        "\nClass mapping:"
    )

    print(
        f"  Original class {VERTICAL_PANEL_CLASS_ID}"
        f" -> New class {TARGET_CLASS_ID}"
        f" (vertical_panel)"
    )

    print(
        "\nAll non-Vertical-Panel annotations "
        "were discarded."
    )

    print(
        "\nSplit:"
    )

    print(
        f"  Train: {TRAIN_RATIO * 100:.0f}%"
    )

    print(
        f"  Test:  {TEST_RATIO * 100:.0f}%"
    )

    print(
        f"  Val:   {VAL_RATIO * 100:.0f}%"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()