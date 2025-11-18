import os
import json
import sqlite3

from aeneas.executetask import ExecuteTask
from aeneas.task import Task

from audio_onset import detect_and_trim

DB_PATH = "db/sentences.db"
AUDIO_DIR = "static/audio"
TRIMMED_AUDIO_DIR = "static/audio_trimmed"   # new directory for trimmed natives
ALIGN_OUTPUT_DIR = "static/json/align"

os.makedirs(TRIMMED_AUDIO_DIR, exist_ok=True)
os.makedirs(ALIGN_OUTPUT_DIR, exist_ok=True)


def run_alignment(audio_path, pinyin_text, output_json_path):
    """
    Run Aeneas alignment for a given audio + pinyin transcript.

    NOTE: This expects `audio_path` to already be onset-normalized
    (i.e., speech starts around 0.0). We do that in `main()` by
    trimming using audio_onset.detect_and_trim.
    """
    # Aeneas text file: one token per line
    with open("temp_pinyin.txt", "w", encoding="utf-8") as f:
        for word in pinyin_text.strip().split():
            f.write(word + "\n")

    config_string = u"task_language=cmn|os_task_file_format=json|is_text_type=plain"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath("temp_pinyin.txt")
    task.sync_map_file_path_absolute = os.path.abspath(output_json_path)

    ExecuteTask(task).execute()
    task.output_sync_map_file()


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT audio_filename, pinyin FROM Sentences")
    rows = cursor.fetchall()
    conn.close()

    for audio_filename, pinyin_text in rows:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        align_json_path = os.path.join(
            ALIGN_OUTPUT_DIR,
            audio_filename.replace(".wav", "_align.json")
        )
        meta_json_path = os.path.join(
            ALIGN_OUTPUT_DIR,
            audio_filename.replace(".wav", "_align_meta.json")
        )

        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio: {audio_path}")
            continue
        if not pinyin_text:
            print(f"[WARN] No pinyin for {audio_filename}")
            continue

        # 1) Make trimmed native audio based on onset detection
        trimmed_audio_path = os.path.join(TRIMMED_AUDIO_DIR, audio_filename)

        try:
            onset_time, _ = detect_and_trim(audio_path, trimmed_audio_path)
        except Exception as e:
            print(f"[ERROR] Onset detection failed for {audio_path}: {e}")
            # Fallback: use original audio
            onset_time = 0.0
            trimmed_audio_path = audio_path

        print(
            f"[INFO] {audio_filename}: "
            f"onset={onset_time:.3f}s, "
            f"running Aeneas on {trimmed_audio_path}"
        )

        # 2) Run Aeneas on the onset-normalized audio
        run_alignment(trimmed_audio_path, pinyin_text, align_json_path)

        # 3) Save meta so you know how this maps back to original audio if needed
        meta = {
            "original_audio": audio_path,
            "trimmed_audio": trimmed_audio_path,
            "onset_time_original_sec": onset_time,
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
