import os
import cv2

# ==========================================================
# Configuration
# ==========================================================
INPUT_TXT = r"C:\temp\clips.txt"
OUTPUT_PATH = r"C:\temp\ExtractedFrames"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# Main
# ==========================================================
with open(INPUT_TXT, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    # Expected format:
    # C:\Videos\clip.mp4 12.5 18.3
    parts = line.rsplit(maxsplit=2)
    if len(parts) != 3:
        print(f"Skipping malformed line:\n{line}")
        continue

    video_path = parts[0]
    start_time = float(parts[1])
    end_time = float(parts[2])
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Invalid FPS for {video_path}")
        cap.release()
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(start_time * fps))
    end_frame = min(total_frames - 1, int(end_time * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {clip_name}")
    frame_id = start_frame
    while frame_id <= end_frame:
        success, frame = cap.read()
        if not success:
            break

        timestamp = frame_id / fps
        filename = (f"{clip_name}_{frame_id:06d}_{timestamp:.3f}.jpg")
        output_file = os.path.join(OUTPUT_PATH, filename)
        cv2.imwrite(output_file, frame)
        frame_id += 1

    cap.release()
print("Done.")