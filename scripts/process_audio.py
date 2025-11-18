import parselmouth
import os
import json

from audio_onset import detect_and_trim

AUDIO_DIR = "static/audio"
TRIMMED_AUDIO_DIR = "static/audio_trimmed"
OUTPUT_DIR = "static/json/pitch"

os.makedirs(TRIMMED_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pitch(audio_path):
    snd = parselmouth.Sound(audio_path)

    pitch_obj = snd.to_pitch(
        time_step=0.01,      # 10 ms frames; match this across your pipeline
        pitch_floor=75,
        pitch_ceiling=500,
    )

    pitch_values = pitch_obj.selected_array["frequency"].tolist()
    dt = pitch_obj.dx
    start_time = pitch_obj.x1

    return pitch_values, dt, start_time


def main():
    for filename in os.listdir(AUDIO_DIR):
        if not filename.endswith(".wav"):
            continue

        base = filename.replace(".wav", "")
        orig_audio_path = os.path.join(AUDIO_DIR, filename)
        trimmed_audio_path = os.path.join(TRIMMED_AUDIO_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{base}_pitch.json")

        if os.path.exists(output_path):
            print(f"[skip] {output_path}")
            continue

        # 1) Trim native audio to onset
        try:
            onset_time, _ = detect_and_trim(orig_audio_path, trimmed_audio_path)
            print(f"[native] {filename}: onset={onset_time:.3f}s")
        except Exception as e:
            print(f"[warn] onset detection failed for {filename}: {e}")
            # fallback: use original
            trimmed_audio_path = orig_audio_path
            onset_time = 0.0

        # 2) Extract pitch from trimmed audio
        pitch_values, dt, start_time = extract_pitch(trimmed_audio_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pitch": pitch_values,
                    "time_step": dt,
                    "start_time": start_time,
                    "original_onset_sec": onset_time,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
