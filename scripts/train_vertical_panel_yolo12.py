from pathlib import Path
import sys

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATASET_YAML = Path(
    r"J:\TCDNet\Roboflow_TCD\data.yaml"
)


# ------------------------------------------------------------
# PRETRAINED MODEL
# ------------------------------------------------------------

MODEL_PATH = "yolo12s.pt"


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

EPOCHS = 100

IMAGE_SIZE = 640

BATCH_SIZE = 8

DEVICE = "cpu"

WORKERS = 4

PATIENCE = 20


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"E:\Projects\patch-convolutional-vision-transformer"
)

RUN_NAME = "vertical_panel_yolo12s"


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO12 Vertical Panel Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if not DATASET_YAML.exists():

        print(
            "\nERROR: Dataset YAML does not exist:"
        )

        print(DATASET_YAML)

        sys.exit(1)

    print("\nDataset:")
    print(DATASET_YAML)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDevice:")
    print(DEVICE)

    print("\nEpochs:")
    print(EPOCHS)

    print("\nImage size:")
    print(IMAGE_SIZE)

    print("\nBatch size:")
    print(BATCH_SIZE)

    # --------------------------------------------------------
    # Load pretrained YOLO12
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Loading pretrained YOLO12")
    print("=" * 70)

    model = YOLO(MODEL_PATH)

    print("\nModel loaded.")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    results = model.train(

        # Dataset
        data=str(DATASET_YAML),

        # Training duration
        epochs=EPOCHS,

        # Input resolution
        imgsz=IMAGE_SIZE,

        # Batch
        batch=BATCH_SIZE,

        # GPU
        device=DEVICE,

        # DataLoader
        workers=WORKERS,

        # Reproducibility
        seed=SEED,

        # ----------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------

        save=True,

        save_period=10,

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val=True,

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        patience=PATIENCE,

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        cache=False,

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        amp=True,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        project=str(PROJECT_DIR),

        name=RUN_NAME,

        exist_ok=True,

        # ----------------------------------------------------
        # Verbose output
        # ----------------------------------------------------

        verbose=True,
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    best_model = (
        PROJECT_DIR
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    last_model = (
        PROJECT_DIR
        / RUN_NAME
        / "weights"
        / "last.pt"
    )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nBest model:")

    print(best_model)

    print("\nLast checkpoint:")

    print(last_model)

    print("\nTraining results:")

    print(
        PROJECT_DIR / RUN_NAME
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()