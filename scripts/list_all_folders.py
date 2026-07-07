import os

# ==========================================================
# Configuration
# ==========================================================
INPUT_FOLDER = r"J:\TCDNet"
OUTPUT_TXT = r"J:\TCDNet\instructions.txt"

# ==========================================================
# Main
# ==========================================================
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        full_path = os.path.join(INPUT_FOLDER, filename)
        if os.path.isfile(full_path):
            f.write(full_path + "\n")

print(f"Saved file list to:\n{OUTPUT_TXT}")