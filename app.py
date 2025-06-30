from flask import Flask, render_template, jsonify, send_from_directory
import os
import json

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
            return jsonify({"error": f"No metdata for {filename}"}), 404
        
        return jsonify({
            "alignment": pitch_data,
            "pinyin": sentence["pinyin"],
            "traditional": sentence["chinese_traditional"],
            "simplified": sentence["chinese_simplified"],
            "audio": sentence["audio_filename"]
        })

    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    
@app.route("/audio/<filename>")
def get_audio_file(filename):
    return send_from_directory("static/audio", filename)

@app.route("/api/available")
def get_available():
    pitch_dir = os.path.join("static", "json", "aligned_pitch")
    files = os.listdir(pitch_dir)
    ids = [f.split("_")[0] for f in files if f.endswith("_aligned_pitch.json")]
    return jsonify(sorted(set(ids)))

if __name__ == "__main__":
    app.run(debug=True)