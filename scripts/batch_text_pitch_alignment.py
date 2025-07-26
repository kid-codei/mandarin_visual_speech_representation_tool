import os
import json 
import parselmouth

AUDIO_DIR = "static/audio"
ALIGN_DIR = "static/json/align"
PITCH_DIR = "static/json/pitch"
OUTPUT_DIR = "static/json/aligned_pitch"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def align_pitch_to_text(filename):
    audio_path = os.path.join(AUDIO_DIR, filename + ".wav")
    pitch_path = os.path.join(PITCH_DIR, filename + "_pitch.json")
    align_path = os.path.join(ALIGN_DIR, filename + "_align.json")

    if not (os.path.exists(audio_path) and os.path.exists(pitch_path) and os.path.exists(align_path)):
        print(f"[skip] Missing file for {filename}")
        return
    
    # get audio dur
    snd = parselmouth.Sound(audio_path)
    duration = snd.get_total_duration()

    with open(pitch_path) as f:
        pitch_values = json.load(f)
    with open(align_path) as f:
        align_data = json.load(f)

    time_step = duration / len(pitch_values)
    aligned = []

    # to handle specific aeneas output that is unexpected
    # load and normalize aeneas alignment json
    with open(align_path) as f:
        raw_align = json.load(f)

    # handle wrapped aeneas structure
    if isinstance(raw_align, dict):
        if "fragments" in raw_align:
            align_data = raw_align["fragments"]
        elif "children" in raw_align:  
            align_data = raw_align["children"]
        else:
            print(f"[skip] Unexpected format in {align_path}")
            return
    elif isinstance(raw_align, list):
        align_data = raw_align
    else:
        print(f"[skip] Could not parse alignment for {filename}")
        return

    for entry in align_data:
        # print(f"entry: {entry} | type: {type(entry)}")
        start = float(entry["begin"])
        end = float(entry["end"])

        text = entry.get("text")

        if text is None and "lines" in entry:
            text = " ".join(entry["lines"])

        if not text:
            continue

        start_idx = max(0, int(start / time_step))
        end_idx = min(len(pitch_values), int(end / time_step))

        pitch_segment = pitch_values[start_idx:end_idx]
        aligned.append({
            "text": text,
            "start": round(start, 3),
            "end": round(end, 3),
            "pitch_segment": pitch_segment
        })

    # dump aligned word and audio info in json
    out_path = os.path.join(OUTPUT_DIR, f"{filename}_aligned_pitch.json")
    with open(out_path, "w") as f:
        json.dump(aligned, f, indent=2)

    print(f"[saved] {out_path}")

def run_batch():
    for filename in os.listdir(ALIGN_DIR):
        if filename.endswith("_align.json"):
            base = filename.replace("_align.json", "")
            align_pitch_to_text(base)

if __name__ == "__main__":
    run_batch()