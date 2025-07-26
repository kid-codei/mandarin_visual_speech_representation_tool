from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
from pydub import AudioSegment
import parselmouth
import traceback

app = Flask(__name__)


# for html front end
@app.route("/")
def index():
    return render_template("index.html")

# show availale sentences with valid json files to support plotting
@app.route("/api/available")
def get_available():
    pitch_dir = os.path.join("static", "json", "aligned_pitch")
    files = [f.split("_")[0] for f in os.listdir(pitch_dir) if f.endswith("_aligned_pitch.json")]
    return jsonify(sorted(set(files)))

# get pitch and alignment data for sentence chosen
@app.route("/api/pitch/<filename>")
def get_pitch_data(filename):
    try:
        pitch_path = os.path.join("static", "json", "aligned_pitch", f"{filename}_aligned_pitch.json")
        with open(pitch_path, "r") as f:
            pitch_data = json.load(f)

        with open(os.path.join("static", "json", "sentences.json"), "r", encoding="utf-8") as f:
            sentences = json.load(f)

        sentence = next((s for s in sentences if s["audio_filename"].split(".")[0] == filename), None)
        if not sentence:
            return jsonify({"error": f"No metadata found for {filename}"}), 404

        return jsonify({
            "alignment": pitch_data,
            "pinyin": sentence["pinyin"],
            "traditional": sentence["chinese_traditional"],
            "simplified": sentence["chinese_simplified"],
            "audio": sentence["audio_filename"],
            "english": sentence["english"]
        })

    except FileNotFoundError:
        return jsonify({"error": "Pitch file not found"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# get audio file chosen
@app.route("/audio/<filename>")
def get_audio(filename):
    return send_from_directory("static/audio", filename)

# upload the audio recording from the learner
@app.route("/upload", methods=["POST"])
def upload_audio():
    audio = request.files.get("audio")
    if not audio:
        return "No audio uploaded", 400

    base = audio.filename.replace("_learner.webm", "")
    wav_filename = f"{base}_learner.wav"
    save_path = os.path.join("static", "learner_audio", wav_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    temp_path = os.path.join("static", "learner_audio", "temp.webm")
    try:
        audio.save(temp_path)
        AudioSegment.from_file(temp_path, format="webm").export(save_path, format="wav")
        os.remove(temp_path)
    except Exception as e:
        traceback.print_exc()
        return f"Audio conversion failed: {str(e)}", 500

    return f"Saved {wav_filename}", 200

# for the converted audio wav files
@app.route("/learner_audio/<filename>")
def serve_learner_audio(filename):
    path = os.path.join("static", "learner_audio", filename)
    if not os.path.exists(path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_from_directory("static/learner_audio", filename)

# get pitch contours from the learner recording
@app.route("/api/learner_pitch/<filename>")
def get_learner_pitch(filename):
    audio_path = os.path.join("static", "learner_audio", f"{filename}_learner.wav")
    if not os.path.exists(audio_path):
        return jsonify({"error": "Learner audio not found"}), 404

    try:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch()
        pitch_values = [
            pitch.get_value_in_frame(i) if pitch.get_value_in_frame(i) > 0 else None
            for i in range(pitch.get_number_of_frames())
        ]
        return jsonify({"filename": filename, "pitch": pitch_values})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pitch extraction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
