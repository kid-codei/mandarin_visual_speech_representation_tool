import os
import json
import parselmouth
from textgrid import TextGrid
import sqlite3

DB_PATH = "db/sentences.db"  # adjust if yours lives elsewhere
AUDIO_DIR = "static/audio"
PITCH_DIR = "static/json/pitch"         # existing pitch JSONs
TEXTGRID_DIR = "static/textgrid"        # MFA output
OUTPUT_DIR = "static/json/aligned_pitch"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_character_labels(basename: str):
    """
    Return a list of Traditional Chinese characters for this audio basename.

    basename examples: "m1", "f3"
    DB stores audio_filename like: "m1.wav", "f3.wav"
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # DB schema:
    #   TABLE Sentences (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     chinese_traditional TEXT,
    #     chinese_simplified  TEXT,
    #     english             TEXT,
    #     audio_filename      TEXT,
    #     pinyin              TEXT,
    #     ...
    #   )
    audio_name = f"{basename}.wav"
    cur.execute(
        "SELECT chinese_traditional FROM Sentences WHERE audio_filename = ?",
        (audio_name,),
    )
    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        return []

    trad_sentence = row[0]  # full Traditional Chinese sentence
    # keep only non-space characters; you can also filter punctuation if you want
    chars = [ch for ch in trad_sentence if not ch.isspace()]
    return chars


def load_pitch_and_dt(basename: str):
    """
    Load pitch array, time step (dt), and start_time from JSON.
    Falls back to legacy behavior if JSON is just a list of values.
    """
    audio_path = os.path.join(AUDIO_DIR, f"{basename}.wav")
    pitch_json_path = os.path.join(PITCH_DIR, f"{basename}_pitch.json")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing audio: {audio_path}")
    if not os.path.exists(pitch_json_path):
        raise FileNotFoundError(f"Missing pitch JSON: {pitch_json_path}")

    with open(pitch_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # New format: {"pitch": [...], "time_step": 0.01, "start_time": ...}
    if isinstance(data, dict):
        pitch_values = data["pitch"]
        dt = data["time_step"]
        start_time = data.get("start_time", 0.0)
    else:
        # (Optional) legacy fallback if any old files remain
        pitch_values = data
        snd = parselmouth.Sound(audio_path)
        duration = snd.duration
        n = len(pitch_values)
        dt = duration / n if n > 0 else 0.01
        start_time = 0.0

    return pitch_values, dt, start_time


def align_pitch_to_textgrid(basename: str, tier_name: str = "words"):
    tg_path = os.path.join(TEXTGRID_DIR, f"{basename}.TextGrid")
    if not os.path.exists(tg_path):
        print(f"[skip] No TextGrid for {basename}")
        return

    try:
        pitch_values, dt, pitch_start = load_pitch_and_dt(basename)
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        return

    # --- get character labels from DB ---
    char_labels = get_character_labels(basename)
    C = len(char_labels)
    if C == 0:
        print(f"[warn] No character labels for {basename}, falling back to raw MFA intervals.")
        # If you want a fallback behavior, you could implement it here.
        return

    tg = TextGrid.fromFile(tg_path)

    # Pick the desired tier (likely "words", but check in Praat)
    tier = None
    for t in tg.tiers:
        if t.name == tier_name:
            tier = t
            break

    if tier is None:
        print(
            f"[warn] Tier '{tier_name}' not found in {tg_path}. "
            f"Available tiers: {[t.name for t in tg.tiers]}"
        )
        return

    # Collect speech intervals (ignore pure silence)
    speech_intervals = [
        interval for interval in tier.intervals
        if interval.mark.strip() != ""
    ]
    N = len(speech_intervals)
    if N == 0:
        print(f"[warn] No non-empty intervals in tier '{tier_name}' for {basename}")
        return

    # If MFA already gives exactly C intervals, we can do 1:1
    if N == C:
        grouped = [[interval] for interval in speech_intervals]
    else:
        # Otherwise, bucket N intervals into C groups by index
        # interval_idx -> char_idx = floor(i * C / N)
        grouped = [[] for _ in range(C)]
        for i, interval in enumerate(speech_intervals):
            char_idx = int(i * C / N)
            if char_idx >= C:
                char_idx = C - 1
            grouped[char_idx].append(interval)

    aligned_segments = []

    for char_idx, intervals in enumerate(grouped):
        if not intervals:
            continue  # should not really happen, but safe

        label = char_labels[char_idx] if char_idx < C else "<extra>"

        # Merge MFA intervals for this character
        start_time = float(min(iv.minTime for iv in intervals))
        end_time = float(max(iv.maxTime for iv in intervals))

        # Map times to pitch indices using Praat's original time grid
        start_idx = int(round((start_time - pitch_start) / dt))
        end_idx = int(round((end_time - pitch_start) / dt))

        # Guard against out-of-range indices
        start_idx = max(0, min(start_idx, len(pitch_values)))
        end_idx = max(start_idx, min(end_idx, len(pitch_values)))

        pitch_segment = pitch_values[start_idx:end_idx]

        aligned_segments.append(
            {
                "text": label,   # <-- actual character (no <unk>)
                "start": start_time,
                "end": end_time,
                "pitch_segment": list(map(float, pitch_segment)),
            }
        )

    out_path = os.path.join(OUTPUT_DIR, f"{basename}_aligned_pitch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aligned_segments, f, ensure_ascii=False, indent=2)

    print(
        f"[aligned] {out_path} "
        f"({len(aligned_segments)} segments for {C} characters, from {N} MFA intervals)"
    )


def run_batch(tier_name: str = "words"):
    for fname in os.listdir(TEXTGRID_DIR):
        if not fname.endswith(".TextGrid"):
            continue
        basename = os.path.splitext(fname)[0]
        align_pitch_to_textgrid(basename, tier_name=tier_name)


if __name__ == "__main__":
    # Change "words" to the tier you want (e.g., "phones") if needed
    run_batch(tier_name="words")
