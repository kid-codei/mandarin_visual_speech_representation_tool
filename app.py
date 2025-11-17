from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
from pydub import AudioSegment
import parselmouth
import traceback

from scripts.alignment import evaluate_pronunciation

app = Flask(__name__)
# Enable CORS so a separate Netlify (or other) frontend can call this API.
# In production you can restrict this to your deployed frontend domain.
CORS(app)


# ---------------------------
# HTML front-end (local dev)
# ---------------------------
@app.route("/")
def index():
    """Serve the main HTML page (used for local Flask-based UI)."""
    return render_template("index.html")


# ---------------------------
# Native sentence data
# ---------------------------
@app.route("/api/available")
def get_available():
    """Return a list of sentence IDs that have aligned pitch JSON.

    These IDs are the base names of files in static/json/aligned_pitch,
    e.g. 'm1', 'f2', etc.
    """
    pitch_dir = os.path.join("static", "json", "aligned_pitch")
    files = [
        f.split("_")[0]
        for f in os.listdir(pitch_dir)
        if f.endswith("_aligned_pitch.json")
    ]
    return jsonify(sorted(set(files)))


@app.route("/api/pitch/<filename>")
def get_pitch_data(filename: str):
    """Return alignment + metadata for a native sample.

    Response:
      {
        "alignment": [...],        # list from <filename>_aligned_pitch.json
        "pinyin": "...",
        "traditional": "...",
        "simplified": "...",
        "audio": "m1.wav",
        "english": "..."
      }
    """
    try:
        pitch_path = os.path.join(
            "static", "json", "aligned_pitch", f"{filename}_aligned_pitch.json"
        )
        with open(pitch_path, "r", encoding="utf-8") as f:
            pitch_data = json.load(f)

        with open(
            os.path.join("static", "json", "sentences.json"), "r", encoding="utf-8"
        ) as f:
            sentences = json.load(f)

        sentence = next(
            (s for s in sentences if s["audio_filename"].split(".")[0] == filename),
            None,
        )
        if not sentence:
            return jsonify({"error": f"No metadata found for {filename}"}), 404

        return jsonify(
            {
                "alignment": pitch_data,
                "pinyin": sentence["pinyin"],
                "traditional": sentence["chinese_traditional"],
                "simplified": sentence["chinese_simplified"],
                "audio": sentence["audio_filename"],
                "english": sentence["english"],
            }
        )

    except FileNotFoundError:
        return jsonify({"error": "Pitch file not found"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Serve native audio files
@app.route("/audio/<filename>")
def get_audio(filename: str):
    """Serve native audio from static/audio."""
    return send_from_directory("static/audio", filename)


# ---------------------------
# Learner recordings
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload_audio():
    """Accept a learner recording uploaded from the browser.

    We accept various containers (webm, mp4, ogg, etc.) and convert to WAV:
      static/learner_audio/<base>_learner.wav
    where <base> matches the sentence ID (e.g. 'm1').
    """
    audio = request.files.get("audio")
    if not audio:
        return "No audio uploaded", 400

    # Strip extension first
    original_name = audio.filename or ""
    name_no_ext, ext = os.path.splitext(original_name)

    # Strip trailing "_learner" if present (works for .webm, .mp4, .ogg, etc.)
    if name_no_ext.endswith("_learner"):
        base = name_no_ext[:-len("_learner")]
    else:
        base = name_no_ext  # fallback

    wav_filename = f"{base}_learner.wav"
    save_dir = os.path.join("static", "learner_audio")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, wav_filename)

    # Determine a reasonable temp extension from mimetype or filename
    mimetype = (audio.mimetype or "").split(";")[0]  # e.g. "audio/webm"
    ext_map = {
        "audio/webm": "webm",
        "video/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/wav": "wav",
    }
    input_ext = ext_map.get(mimetype, ext.lstrip(".") or "webm")

    temp_path = os.path.join(save_dir, f"temp.{input_ext}")

    try:
        audio.save(temp_path)

        # If the file is empty, bail early with a clearer error
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return "Empty recording uploaded; please try again.", 400

        # Let pydub/ffmpeg auto-detect input format based on file header
        sound = AudioSegment.from_file(temp_path)
        sound.export(save_path, format="wav")

        os.remove(temp_path)
    except Exception as e:
        traceback.print_exc()
        # Clean up temp file if it exists
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return f"Audio conversion failed: {str(e)}", 500

    return f"Saved {wav_filename}", 200



@app.route("/learner_audio/<filename>")
def serve_learner_audio(filename: str):
    """Serve a learner WAV recording from static/learner_audio."""
    path = os.path.join("static", "learner_audio", filename)
    if not os.path.exists(path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_from_directory("static/learner_audio", filename)


@app.route("/api/learner_pitch/<filename>")
def get_learner_pitch(filename: str):
    """Return raw learner pitch contour for plotting.

    Response:
      {
        "filename": "m1",
        "pitch": [ ... ]   # Hz values (0 or null for unvoiced)
      }
    """
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


# ---------------------------
# Pronunciation evaluation
# ---------------------------
@app.route("/api/evaluate/<filename>")
def evaluate(filename: str):
    """Evaluate learner pronunciation for a given sentence.

    This assumes a learner WAV exists at:
      static/learner_audio/<filename>_learner.wav

    Response:
      {
        "overall_score": 78.5,
        "segments": [
          {
            "index": 0,
            "text": "你",
            "start": 0.12,
            "end": 0.53,
            "score": 82.1,
            "color": "green",
            "pitch_direction": "up"
          },
          ...
        ]
      }
    """
    try:
        learner_path = os.path.join(
            "static", "learner_audio", f"{filename}_learner.wav"
        )
        result = evaluate_pronunciation(filename, learner_path)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Local dev: python app.py
    # For deployment platforms (Render, Railway, etc.), respect $PORT.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)

