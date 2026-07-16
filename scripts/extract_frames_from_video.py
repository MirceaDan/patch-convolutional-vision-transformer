import os
import cv2

# ==========================================================
# Configuration
# ==========================================================

INPUT_TXT = r"J:\TCDNet\instructions.txt"

OUTPUT_PATH = r"J:\TCDNet\ExtractedFrames"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==========================================================
# Helper functions
# ==========================================================

def timestamp_to_frame(timestamp: str, fps: float) -> int:
    """
    Converts MM:SS:FF to absolute frame number.
    FF = frame inside the current second.
    """
    minutes, seconds, frame = map(int, timestamp.split(":"))
    return int((minutes * 60 + seconds) * fps + frame)


def frame_to_timestamp(frame_id: int, fps: float) -> str:
    """
    Converts absolute frame number back to MM-SS-FF.
    """
    total_seconds = int(frame_id // fps)

    minutes = total_seconds // 60
    seconds = total_seconds % 60
    frame = int(round(frame_id - total_seconds * fps))

    # avoid rounding problems (e.g. frame==30)
    if frame >= int(round(fps)):
        frame = 0
        seconds += 1

        if seconds >= 60:
            seconds = 0
            minutes += 1

    return f"{minutes:02d}-{seconds:02d}-{frame:02d}"


# ==========================================================
# Main
# ==========================================================

with open(INPUT_TXT, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Found {len(lines)} intervals.\n")

for idx, line in enumerate(lines, start=1):

    parts = line.rsplit(maxsplit=2)

    if len(parts) != 3:
        print(f"[{idx}] Invalid line:")
        print(line)
        continue

    video_path = parts[0]
    start_timestamp = parts[1]
    end_timestamp = parts[2]

    if not os.path.exists(video_path):
        print(f"[{idx}] Video not found:")
        print(video_path)
        continue

    print(f"[{idx}/{len(lines)}] Processing:")
    print(os.path.basename(video_path))
    print(f"Interval: {start_timestamp} -> {end_timestamp}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.\n")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Invalid FPS.\n")
        cap.release()
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = timestamp_to_frame(start_timestamp, fps)
    end_frame = timestamp_to_frame(end_timestamp, fps)

    if start_frame >= total_frames:
        print("Start frame outside video.\n")
        cap.release()
        continue

    end_frame = min(end_frame, total_frames - 1)

    if end_frame < start_frame:
        print("Invalid interval.\n")
        cap.release()
        continue

    clip_name = os.path.splitext(os.path.basename(video_path))[0]

    clip_output_folder = os.path.join(OUTPUT_PATH, clip_name)
    os.makedirs(clip_output_folder, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_id = start_frame

    saved = 0

    while frame_id <= end_frame:

        success, frame = cap.read()

        if not success:
            break

        timestamp = frame_to_timestamp(frame_id, fps)

        filename = (
            f"{clip_name}_{frame_id:06d}_{timestamp}.jpg"
        )

        output_file = os.path.join(
            clip_output_folder,
            filename
        )

        cv2.imwrite(output_file, frame)

        saved += 1
        frame_id += 1

    cap.release()

    print(f"Saved {saved} frames.\n")

print("===================================")
print("Finished extracting all intervals.")
print("===================================")