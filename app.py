from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
import parselmouth
from pydub import AudioSegment
import traceback
from math import isnan  

app = Flask(__name__)

# set route for frontend
@app.route("/")
def index():
    return render_template("index.html")

# api route to get aligned pitch data
@app.route("/api/pitch/<filename>")
def get_pitch_data(filename):
    try:
        pitch_path = os.path.join("static", "json", "aligned_pitch", f"{filename}_aligned_pitch.json")
        print("path name:", pitch_path)
        with open(pitch_path, "r") as f:
            pitch_data = json.load(f)

        sentence_path = os.path.join("static", "json", "sentences.json")
        with open(sentence_path, "r", encoding="utf-8") as f:
            all_sentences = json.load(f)

        sentence = next(
            (s for s in all_sentences if s["audio_filename"].split(".")[0] == filename),
            None
        )

        if sentence is None:
            return jsonify({"error": f"No metadata for {filename}"}), 404
        
        return jsonify({
            "alignment": pitch_data,
            "pinyin": sentence["pinyin"],
            "traditional": sentence["chinese_traditional"],
            "simplified": sentence["chinese_simplified"],
            "audio": sentence["audio_filename"],
            "english": sentence["english"]
        })

    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# route to fetch the uploaded audio file
@app.route("/audio/<filename>")
def get_audio_file(filename):
    return send_from_directory("static/audio", filename)

# api route to get available sentences
@app.route("/api/available")
def get_available():
    pitch_dir = os.path.join("static", "json", "aligned_pitch")
    files = os.listdir(pitch_dir)
    ids = [f.split("_")[0] for f in files if f.endswith("_aligned_pitch.json")]
    return jsonify(sorted(set(ids)))

# route to handle the audio upload
@app.route("/upload", methods=["POST"])
def upload_audio():
    audio = request.files["audio"]
    if not audio:
        return "No audio uploaded", 400

    filename = audio.filename  # e.g. f1_learner.webm
    base = filename.replace("_learner.webm", "")
    wav_filename = f"{base}_learner.wav"
    save_path = os.path.join("static/learner_audio", wav_filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # save WebM temporarily
    temp_path = os.path.join("static/learner_audio", "temp.webm")
    audio.save(temp_path)

    # convert to WAV
    sound = AudioSegment.from_file(temp_path, format="webm")
    sound.export(save_path, format="wav")

    # clean up temp file
    os.remove(temp_path)

    return f"Saved {wav_filename}", 200

# route to serve learner audio
@app.route("/learner_audio/<filename>")
def get_learner_audio(filename):
    try:
        # make sure file exists before attempting to send it
        if not os.path.exists(os.path.join("static", "learner_audio", filename)):
            return jsonify({"error": "Audio file not found"}), 404
        return send_from_directory("static/learner_audio", filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# api route to process the learner pitch
@app.route("/api/learner_pitch/<filename>")
def get_learner_pitch(filename):
    audio_path = os.path.join("static", "learner_audio", f"{filename}_learner.wav")
    
    print("Looking for:", audio_path)
    if not os.path.exists(audio_path):
        return jsonify({"error": "Learner audio not found"}), 404

    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = []

    for t in range(pitch.get_number_of_frames()):
        hz = pitch.get_value_in_frame(t)
        pitch_values.append(hz if hz > 0 else None)  # None = unvoiced

    return jsonify({
        "filename": filename,
        "pitch": pitch_values
    })

if __name__ == "__main__":
    app.run(debug=True)
