import os
import json
import argparse
import matplotlib.pyplot as plt
import parselmouth

AUDIO_DIR = "static/audio"
PITCH_DIR = "static/json/pitch"
OUTPUT_DIR = "static/json/plots"

def plot_pitch(filename):
    audio_path = os.path.join(AUDIO_DIR, filename + ".wav")
    json_path = os.path.join(PITCH_DIR, filename + "_pitch.json")

    if not os.path.exists(audio_path) or not os.path.exists(json_path):
        print(f"Missing audio or pitch JSON for {filename}")
        return 
    
    snd = parselmouth.Sound(audio_path)
    duration = snd.get_total_duration()

    with open(json_path, "r") as f:
        pitch_values = json.load(f)

    time_step = duration/len(pitch_values)
    times = [i * time_step for i in range(len(pitch_values))]

    # plot audio
    plt.figure(figsize=(10,4))
    plt.plot(times, pitch_values, label="Pitch (Hz)", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Pitch Contour: {filename}")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{filename}_pitch_plot.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Filename without extension (e.g., 001)")
    args = parser.parse_args()
    plot_pitch(args.filename)