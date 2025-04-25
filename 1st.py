#for data extraction from the video


import os
import glob
import json
import cv2
import re
from datetime import datetime, timedelta
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────
JSON_PATH  = r"D:/videos/book1.json"
VIDEO_DIR  = r"D:/videos"
OUTPUT_DIR = r"D:/videos/output_images"
DATE_FMT   = "%d-%b-%Y %H:%M:%S"   # format of your JSON "Transaction DateTime"

# Fixed timestamp offset: JSON timestamps are 1:44 (104 seconds) ahead of video timing
TIMESTAMP_OFFSET = 104  # 1 minute and 44 seconds in seconds

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────

def sanitize_simple(s: str) -> str:
    """
    Replace non-alphanumeric runs with single underscore and strip edges
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def sanitize_material(raw: str) -> str:
    """
    Normalize the material field, preserving semicolon at end
    """
    has_semicolon = raw.rstrip().endswith(";")
    # remove semicolons and literal ":-"
    t = raw.replace(";", "").replace(":-", "")
    # collapse whitespace
    t = re.sub(r"\s+", "_", t.strip())
    if has_semicolon:
        t += ";"
    return t


def extract_datetime_from_filename(filename):
    """Extract start and end times from video filename"""
    parts = filename.split('_')
    if len(parts) >= 5:
        try:
            start_time_str = parts[3]
            end_time_str = parts[4].split('.')[0]
            start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")
            end_time   = datetime.strptime(end_time_str, "%Y%m%d%H%M%S")
            return start_time, end_time
        except (ValueError, IndexError):
            pass
    return None, None


def get_video_duration(video_path):
    """Get the actual duration of a video file in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (frame_count / fps) if fps > 0 else 0


def create_video_map():
    video_map = []
    video_paths = glob.glob(os.path.join(VIDEO_DIR, "*steel yard_ch1_main*"))
    print("Analyzing video files...")
    for vp in tqdm(sorted(video_paths)):
        filename = os.path.basename(vp)
        nominal_start, nominal_end = extract_datetime_from_filename(filename)
        if nominal_start and nominal_end:
            actual_duration = get_video_duration(vp)
            video_map.append({
                'path': vp,
                'nominal_start': nominal_start,
                'nominal_end': nominal_end,
                'actual_duration': actual_duration
            })
    video_map.sort(key=lambda x: x['nominal_start'])
    return video_map


def find_video_for_timestamp(ts, video_map):
    video_ts = ts - timedelta(seconds=TIMESTAMP_OFFSET)
    for video in video_map:
        if video['nominal_start'] <= video_ts < video['nominal_end']:
            return video
    return None


def extract_frame_for_transaction(entry, video_map):
    ts = datetime.strptime(entry["Transaction DateTime"], DATE_FMT)
    video = find_video_for_timestamp(ts, video_map)
    if not video:
        return False, f"No video for {ts} (adjusted {ts - timedelta(seconds=TIMESTAMP_OFFSET)})"

    video_ts = ts - timedelta(seconds=TIMESTAMP_OFFSET)
    delta_seconds = (video_ts - video['nominal_start']).total_seconds()
    if delta_seconds < 0:
        return False, f"Adjusted timestamp {video_ts} before video start"
    if delta_seconds > video['actual_duration']:
        return False, f"Adjusted timestamp {video_ts} beyond duration {video['actual_duration']:.1f}s"

    cap = cv2.VideoCapture(video['path'])
    if not cap.isOpened():
        return False, f"Cannot open {os.path.basename(video['path'])}"

    cap.set(cv2.CAP_PROP_POS_MSEC, delta_seconds * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False, f"Failed to read frame at {delta_seconds:.1f}s"

    # build custom filename
    vt  = sanitize_simple(entry.get("Vehicle Type", ""))
    wt  = sanitize_simple(entry.get("Weighment Type", ""))
    w   = sanitize_simple(str(entry.get("Weight", "")))
    mat = sanitize_material(entry.get("Material", ""))
    ts_part = ts.strftime("%Y%m%d_%H%M%S")
    out_name = f"{vt}_{wt}_{w}_{mat}_{ts_part}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, frame)

    return True, f"Saved {out_name}"


def main():
    print(f"Loading transactions from {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    print(f"Loaded {len(transactions)} transactions")

    video_map = create_video_map()
    print(f"Found {len(video_map)} video files")

    offset_min = TIMESTAMP_OFFSET // 60
    offset_sec = TIMESTAMP_OFFSET % 60
    print(f"Timestamps are ahead by {offset_min}m {offset_sec}s")

    success = 0
    print("Extracting frames...")
    for entry in tqdm(transactions):
        ok, msg = extract_frame_for_transaction(entry, video_map)
        if ok:
            success += 1
            tqdm.write(f"✅ {msg}")
        else:
            tqdm.write(f"⚠️ {msg}")

    print(f"Done: {success}/{len(transactions)} images saved.")

if __name__ == "__main__":
    main()
