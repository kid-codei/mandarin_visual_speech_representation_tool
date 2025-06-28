import parselmouth
import os
import json

AUDIO_DIR = "static/audio"
OUTPUT_DIR = "static/json/pitch"

def extract_pitch(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"].tolist()
    return pitch_values

def main():
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav"):
            base = filename.replace(".wav", "")
            audio_path = os.path.join(AUDIO_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"{base}_pitch.json")

            if os.path.exists(output_path):
                print(f"[skip] {output_path}")
                continue

            pitch = extract_pitch(audio_path)
            with open(output_path, "w") as f:
                json.dump(pitch, f)
            print(f"[saved] {output_path}")

if __name__ == "__main__":
    main()