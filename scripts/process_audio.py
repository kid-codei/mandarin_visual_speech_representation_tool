import parselmouth
import os
import json

AUDIO_DIR = "static/audio"
OUTPUT_DIR = "static/json/pitch"

# get pitch values from audio file
# get pitch values from audio file
def extract_pitch(audio_path):
    snd = parselmouth.Sound(audio_path)

    # Use consistent settings for Praat pitch extraction
    pitch_obj = snd.to_pitch(
        time_step=0.01,      # 10 ms frames; keep this consistent everywhere
        pitch_floor=75,
        pitch_ceiling=500,
    )

    # Hz values
    pitch_values = pitch_obj.selected_array["frequency"].tolist()

    # Exact time grid used by Praat:
    #   dx = time step between frames
    #   x1 = time of first frame
    dt = pitch_obj.dx
    start_time = pitch_obj.x1

    return pitch_values, dt, start_time



# handle finding audio file and where to store
# get pitches and dump info into json
def main():
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav"):
            base = filename.replace(".wav", "")
            audio_path = os.path.join(AUDIO_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"{base}_pitch.json")

            if os.path.exists(output_path):
                print(f"[skip] {output_path}")
                continue

            pitch_values, dt, start_time = extract_pitch(audio_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pitch": pitch_values,
                        "time_step": dt,
                        "start_time": start_time,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[saved] {output_path}")



if __name__ == "__main__":
    main()