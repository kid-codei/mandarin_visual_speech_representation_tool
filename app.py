from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
from pydub import AudioSegment
import parselmouth
import traceback

from scripts.alignment import evaluate_pronunciation
from scripts.audio_onset import detect_and_trim
from scripts.db_log import log_attempt, get_history

app = Flask(__name__, static_folder="frontend", static_url_path="")
# Enable CORS so a separate Netlify (or other) frontend can call this API.
# In production you can restrict this to your deployed frontend domain.
CORS(app)


# ---------------------------
# HTML front-end (local dev)
# ---------------------------
@app.route("/")
def intro():
    return send_from_directory("frontend", "index.html")


@app.route("/index.html")
def tool():
    return send_from_directory("frontend", "tool.html")



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

    We expect form-data:
      - audio: the recorded blob
      - sentence_id: e.g. "f4", "m1"

    We save as:
      static/learner_audio/<sentence_id>_learner.wav
    Then we trim silence using scripts.audio_onset.detect_and_trim.
    """
    print("\n[upload] ===== New upload request =====")
    print("[upload] CWD:", os.getcwd())

    audio = request.files.get("audio")
    sentence_id = request.form.get("sentence_id") or request.args.get("sentence_id")

    print("[upload] form sentence_id:", sentence_id)
    print("[upload] file object:", audio)

    if not audio:
        return "No audio uploaded", 400
    if not sentence_id:
        return "Missing sentence_id", 400

    base = sentence_id.strip()
    if not base:
        return "Invalid sentence_id", 400

    wav_filename = f"{base}_learner.wav"
    save_dir = os.path.join("static", "learner_audio")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, wav_filename)

    print("[upload] original_name:", audio.filename)
    print("[upload] target path:", save_path)

    # Determine temp container extension from mimetype or filename
    mimetype = (audio.mimetype or "").split(";")[0]  # e.g. "audio/webm"
    ext_map = {
        "audio/webm": "webm",
        "video/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/wav": "wav",
    }
    _, ext = os.path.splitext(audio.filename or "")
    input_ext = ext_map.get(mimetype, ext.lstrip(".") or "webm")

    temp_path = os.path.join(save_dir, f"temp.{input_ext}")
    print("[upload] temp_path:", temp_path)

    try:
        audio.save(temp_path)
        print("[upload] saved temp, size:", os.path.getsize(temp_path))

        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return "Empty recording uploaded; please try again.", 400

        # Convert to WAV using pydub/ffmpeg
        sound = AudioSegment.from_file(temp_path)
        sound.export(save_path, format="wav")
        os.remove(temp_path)
        print("[upload] exported WAV to:", save_path)

        # 🔹 Trim onset using your scripts/audio_onset.py
        # Assumed signature: detect_and_trim(input_wav, output_wav) -> (onset_time, output_path)
        onset_time, trimmed_path = detect_and_trim(
            save_path,
            save_path + ".tmp.wav",
        )
        os.replace(trimmed_path, save_path)
        print(f"[upload] trimmed onset={onset_time:.3f}s, final:", save_path)
        print("[upload] exists after trim?", os.path.exists(save_path))

        return f"Saved {wav_filename}", 200

    except Exception as e:
        traceback.print_exc()
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return f"Audio conversion failed: {str(e)}", 500


@app.route("/learner_audio/<filename>")
def serve_learner_audio(filename: str):
    """Serve a learner WAV recording from static/learner_audio."""
    path = os.path.join("static", "learner_audio", filename)
    if not os.path.exists(path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_from_directory("static/learner_audio", filename)


@app.route("/api/learner_pitch/<filename>")
def get_learner_pitch(filename: str):
    """Return raw learner pitch contour + timing for plotting.

    Response:
      {
        "filename": "f4",
        "pitch": [...],        # Hz (0 or null for unvoiced)
        "time_step": 0.01,     # seconds between frames
        "start_time": 0.0      # time of first frame in seconds
      }
    """
    audio_path = os.path.join("static", "learner_audio", f"{filename}_learner.wav")
    if not os.path.exists(audio_path):
        return jsonify({"error": f"Learner audio not found for {filename}: {audio_path}"}), 404

    try:
        snd = parselmouth.Sound(audio_path)

        # Use same settings as process_audio.py
        pitch_obj = snd.to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=500,
        )

        freqs = pitch_obj.selected_array["frequency"].tolist()
        dt = pitch_obj.dx
        start_time = pitch_obj.x1

        # For plotting we can keep 0.0 for unvoiced or use null; your
        # normalizePitchToSemitones already treats 0/null as unvoiced.
        pitch_values = [f if f > 0 else 0.0 for f in freqs]

        return jsonify({
            "filename": filename,
            "pitch": pitch_values,
            "time_step": dt,
            "start_time": start_time,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pitch extraction failed: {str(e)}"}), 500



# ---------------------------
# Pronunciation evaluation
# ---------------------------
# @app.route("/api/evaluate/<filename>")
@app.route("/api/evaluate/<sentence_id>")
def evaluate(sentence_id):
    """
    Evaluate pronunciation for a given sentence.

    Optional query parameter:
      mode = "normal" (default), "lenient", or "strict"
    """
    mode = request.args.get("mode", "normal")  # <-- NEW

    try:
        result = evaluate_pronunciation(sentence_id, mode=mode)
        try:
            log_attempt(
                sentence_id=sentence_id,
                mode=mode,
                overall_score=result["overall_score"],
                segments=result["segments"],
            )
        except Exception as log_err:
            print(f"[db_log] Failed to persist attempt: {log_err}")
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500


@app.route("/api/history/<sentence_id>")
def history(sentence_id):
    """Return past attempts for a sentence from the persistent DB."""
    try:
        attempts = get_history(sentence_id)
        return jsonify(attempts)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # Local dev: python app.py
    # For deployment platforms (Render, Railway, etc.), respect $PORT.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)

