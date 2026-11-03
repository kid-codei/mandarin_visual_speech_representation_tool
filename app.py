from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
import shutil
import subprocess
import tempfile
from pydub import AudioSegment
import parselmouth
import traceback

from scripts.alignment import evaluate_pronunciation
from scripts.audio_onset import detect_and_trim
from scripts.db_log import log_attempt, get_history

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MFA_DICTIONARY = os.path.join(PROJECT_ROOT, "mfa_resources", "mandarin_pinyin.dict")
MFA_MODEL = os.path.join(PROJECT_ROOT, "mfa_resources", "mandarin_mfa.zip")


# ---------------------------
# HTML front-end
# ---------------------------
@app.route("/")
def intro():
    return send_from_directory("frontend", "index.html")


@app.route("/tool")
@app.route("/tool.html")
def tool():
    return send_from_directory("frontend", "tool.html")


# ---------------------------
# Native sentence data
# ---------------------------
@app.route("/api/available")
def get_available():
    """Return a list of sentence IDs that have aligned pitch JSON."""
    pitch_dir = os.path.join("static", "json", "aligned_pitch")
    files = [
        f.split("_")[0]
        for f in os.listdir(pitch_dir)
        if f.endswith("_aligned_pitch.json")
    ]
    return jsonify(sorted(set(files)))


@app.route("/api/pitch/<filename>")
def get_pitch_data(filename: str):
    """Return alignment + metadata for a native sample."""
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


@app.route("/audio/<filename>")
def get_audio(filename: str):
    return send_from_directory("static/audio", filename)


# ---------------------------
# Learner recordings
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload_audio():
    """Accept a learner recording and save/trim it."""
    print("\n[upload] ===== New upload request =====")

    audio = request.files.get("audio")
    sentence_id = request.form.get("sentence_id") or request.args.get("sentence_id")

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

    mimetype = (audio.mimetype or "").split(";")[0]
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

    try:
        audio.save(temp_path)

        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return "Empty recording uploaded; please try again.", 400

        sound = AudioSegment.from_file(temp_path)
        sound.export(save_path, format="wav")
        os.remove(temp_path)

        onset_time, trimmed_path = detect_and_trim(save_path, save_path + ".tmp.wav")
        os.replace(trimmed_path, save_path)
        print(f"[upload] trimmed onset={onset_time:.3f}s → {save_path}")

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
    path = os.path.join("static", "learner_audio", filename)
    if not os.path.exists(path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_from_directory("static/learner_audio", filename)


@app.route("/api/learner_pitch/<filename>")
def get_learner_pitch(filename: str):
    """Return raw learner pitch contour for plotting."""
    audio_path = os.path.join("static", "learner_audio", f"{filename}_learner.wav")
    if not os.path.exists(audio_path):
        return jsonify({"error": f"Learner audio not found for {filename}"}), 404

    try:
        snd = parselmouth.Sound(audio_path)
        pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        freqs = pitch_obj.selected_array["frequency"].tolist()
        pitch_values = [f if f > 0 else 0.0 for f in freqs]

        return jsonify({
            "filename": filename,
            "pitch": pitch_values,
            "time_step": pitch_obj.dx,
            "start_time": pitch_obj.x1,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pitch extraction failed: {str(e)}"}), 500


# ---------------------------
# MFA alignment
# ---------------------------
@app.route("/api/mfa_align/<sentence_id>", methods=["POST"])
def mfa_align_learner(sentence_id: str):
    """
    Run MFA forced alignment on a learner's uploaded recording.

    Requires MFA to be installed:
        conda install -c conda-forge montreal-forced-aligner

    Steps:
      1. Load pinyin transcript for the sentence.
      2. Build a temporary MFA corpus (WAV + .lab file).
      3. Run `mfa align`.
      4. Parse the resulting TextGrid.
      5. Return segment boundaries as JSON.

    Returns:
      {
        "sentence_id": "f1",
        "segments": [
          {"text": "你", "start": 0.12, "end": 0.35},
          ...
        ]
      }
    """
    learner_wav = os.path.join("static", "learner_audio", f"{sentence_id}_learner.wav")
    if not os.path.exists(learner_wav):
        return jsonify({"error": f"No learner audio found for {sentence_id}. Upload a recording first."}), 404

    # Load pinyin transcript
    sentences_path = os.path.join("static", "json", "sentences.json")
    try:
        with open(sentences_path, "r", encoding="utf-8") as f:
            sentences = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "sentences.json not found"}), 500

    sentence = next(
        (s for s in sentences if s["audio_filename"].split(".")[0] == sentence_id),
        None,
    )
    if not sentence:
        return jsonify({"error": f"No metadata for sentence {sentence_id}"}), 404

    pinyin = sentence.get("pinyin", "").strip()
    if not pinyin:
        return jsonify({"error": "No pinyin transcript found for this sentence"}), 400

    # Check MFA availability
    if not shutil.which("mfa"):
        return jsonify({
            "error": "MFA (Montreal Forced Aligner) is not installed or not in PATH. "
                     "Install with: conda install -c conda-forge montreal-forced-aligner"
        }), 503

    # Build temp corpus
    tmp_corpus = tempfile.mkdtemp(prefix="mfa_corpus_")
    tmp_output = tempfile.mkdtemp(prefix="mfa_output_")

    try:
        corpus_wav = os.path.join(tmp_corpus, f"{sentence_id}.wav")
        corpus_lab = os.path.join(tmp_corpus, f"{sentence_id}.lab")

        shutil.copy2(learner_wav, corpus_wav)
        with open(corpus_lab, "w", encoding="utf-8") as f:
            f.write(pinyin)

        cmd = [
            "mfa", "align",
            tmp_corpus,
            MFA_DICTIONARY,
            MFA_MODEL,
            tmp_output,
            "--clean",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return jsonify({
                "error": "MFA alignment failed",
                "details": result.stderr[-2000:] if result.stderr else "No details"
            }), 500

        # Parse TextGrid output
        tg_path = os.path.join(tmp_output, f"{sentence_id}.TextGrid")
        if not os.path.exists(tg_path):
            return jsonify({"error": "MFA ran but produced no TextGrid output"}), 500

        try:
            from textgrid import TextGrid
        except ImportError:
            return jsonify({"error": "textgrid Python package not installed"}), 500

        tg = TextGrid.fromFile(tg_path)
        words_tier = next((t for t in tg.tiers if t.name == "words"), None)
        if words_tier is None:
            return jsonify({"error": "No 'words' tier found in TextGrid"}), 500

        segments = [
            {"text": iv.mark.strip(), "start": float(iv.minTime), "end": float(iv.maxTime)}
            for iv in words_tier.intervals
            if iv.mark.strip()
        ]

        # Also save TextGrid to project for future use
        out_tg_dir = os.path.join("static", "textgrid", "learner")
        os.makedirs(out_tg_dir, exist_ok=True)
        shutil.copy2(tg_path, os.path.join(out_tg_dir, f"{sentence_id}_learner.TextGrid"))

        return jsonify({"sentence_id": sentence_id, "segments": segments})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "MFA alignment timed out (>120 s)"}), 504
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"MFA alignment error: {str(e)}"}), 500
    finally:
        shutil.rmtree(tmp_corpus, ignore_errors=True)
        shutil.rmtree(tmp_output, ignore_errors=True)


@app.route("/api/mfa_status")
def mfa_status():
    """Check whether MFA is available on this server."""
    available = bool(shutil.which("mfa"))
    return jsonify({"mfa_available": available})


# ---------------------------
# Pronunciation evaluation
# ---------------------------
@app.route("/api/evaluate/<sentence_id>")
def evaluate(sentence_id):
    """Evaluate pronunciation for a given sentence."""
    mode = request.args.get("mode", "normal")

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
