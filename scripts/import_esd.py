"""Import first sentence from each ESD speaker into the app.

For each of the 20 ESD speakers:
  1. Read first line of their transcript file
  2. Copy Neutral audio to static/audio/esd_{speaker}.wav
  3. Extract pitch
  4. Run aeneas alignment (per-character, using pinyin)
  5. Build aligned_pitch.json
  6. Update static/json/sentences.json and the SQLite DB
"""

import json
import os
import shutil
import sqlite3
import sys

import parselmouth
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import opencc
from pypinyin import pinyin, Style

ESD_ROOT = "/Users/johnhenrycruz/Documents/drwu/Emotion Speech Dataset"
STATIC_AUDIO = "static/audio"
PITCH_DIR = "static/json/pitch"
ALIGN_DIR = "static/json/align"
ALIGNED_PITCH_DIR = "static/json/aligned_pitch"
SENTENCES_JSON = "static/json/sentences.json"
DB_PATH = "db/sentences.db"

converter = opencc.OpenCC("s2t")  # simplified -> traditional


def get_pinyin(text: str) -> str:
    """Return space-separated pinyin with tone numbers, skipping punctuation."""
    result = []
    for char, syllable in zip(text, pinyin(text, style=Style.TONE3, heteronym=False)):
        if '一' <= char <= '鿿':
            result.append(syllable[0])
    return " ".join(result)


def extract_pitch(audio_path: str) -> list:
    snd = parselmouth.Sound(audio_path)
    pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    return pitch_obj.selected_array["frequency"].tolist()


def run_aeneas(audio_path: str, text_chars: list, out_json: str):
    """Run aeneas forced alignment on individual characters."""
    # Write a temp text file with one character per line
    tmp_txt = out_json + ".tmp.txt"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        for ch in text_chars:
            f.write(ch + "\n")

    config_str = (
        "task_language=cmn|"
        "is_text_type=plain|"
        "os_task_file_format=json"
    )
    task = Task(config_string=config_str)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(tmp_txt)
    task.sync_map_file_path_absolute = os.path.abspath(out_json)

    ExecuteTask(task).execute()
    task.output_sync_map_file()

    os.remove(tmp_txt)


def build_aligned_pitch(pitch_values: list, align_json_path: str, audio_path: str) -> list:
    snd = parselmouth.Sound(audio_path)
    duration = snd.get_total_duration()
    time_step = duration / len(pitch_values) if pitch_values else 0.01

    with open(align_json_path, encoding="utf-8") as f:
        raw = json.load(f)

    fragments = raw.get("fragments", [])
    aligned = []
    for entry in fragments:
        start = float(entry["begin"])
        end = float(entry["end"])
        text = " ".join(entry.get("lines", [])).strip()
        if not text:
            continue
        start_idx = max(0, int(start / time_step))
        end_idx = min(len(pitch_values), int(end / time_step))
        aligned.append({
            "text": text,
            "start": round(start, 3),
            "end": round(end, 3),
            "pitch_segment": pitch_values[start_idx:end_idx],
        })
    return aligned


def load_sentences_json() -> list:
    if os.path.exists(SENTENCES_JSON):
        with open(SENTENCES_JSON, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_sentences_json(sentences: list):
    with open(SENTENCES_JSON, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)


def upsert_db(sentence_id, simplified, traditional, english, pinyin_str, audio_filename):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM Sentences WHERE audio_filename = ?", (audio_filename,)
    )
    if cur.fetchone():
        cur.execute(
            """UPDATE Sentences
               SET chinese_traditional=?, chinese_simplified=?, english=?, pinyin=?
               WHERE audio_filename=?""",
            (traditional, simplified, english, pinyin_str, audio_filename),
        )
    else:
        cur.execute(
            """INSERT INTO Sentences
               (chinese_traditional, chinese_simplified, english, pinyin, audio_filename)
               VALUES (?,?,?,?,?)""",
            (traditional, simplified, english, pinyin_str, audio_filename),
        )
    conn.commit()
    conn.close()


def main():
    os.makedirs(PITCH_DIR, exist_ok=True)
    os.makedirs(ALIGN_DIR, exist_ok=True)
    os.makedirs(ALIGNED_PITCH_DIR, exist_ok=True)

    speakers = sorted(
        d for d in os.listdir(ESD_ROOT) if os.path.isdir(os.path.join(ESD_ROOT, d))
    )

    sentences = load_sentences_json()
    existing_ids = {s["audio_filename"].split(".")[0] for s in sentences}

    for speaker in speakers:
        speaker_dir = os.path.join(ESD_ROOT, speaker)
        transcript_path = os.path.join(speaker_dir, f"{speaker}.txt")
        if not os.path.exists(transcript_path):
            print(f"[skip] no transcript for {speaker}")
            continue

        with open(transcript_path, encoding="utf-8") as f:
            first_line = f.readline().strip()

        if not first_line:
            print(f"[skip] empty transcript for {speaker}")
            continue

        parts = first_line.split("\t")
        if len(parts) < 2:
            print(f"[skip] bad transcript line for {speaker}: {first_line!r}")
            continue

        utt_id = parts[0]       # e.g. 0001_000001
        simplified = parts[1]   # Chinese simplified text

        # Skip English speakers (0011–0020 in ESD are English)
        if not any('一' <= ch <= '鿿' for ch in simplified):
            print(f"[skip] English speaker: {speaker}")
            continue

        sentence_id = f"esd_{speaker}"
        audio_dest = os.path.join(STATIC_AUDIO, f"{sentence_id}.wav")

        # Find Neutral audio
        neutral_dir = os.path.join(speaker_dir, "Neutral")
        expected_wav = os.path.join(neutral_dir, f"{utt_id}.wav")
        if not os.path.exists(expected_wav):
            # fallback: first wav in Neutral dir
            wavs = sorted(f for f in os.listdir(neutral_dir) if f.endswith(".wav"))
            if not wavs:
                print(f"[skip] no neutral audio for {speaker}")
                continue
            expected_wav = os.path.join(neutral_dir, wavs[0])
            print(f"[warn] using fallback audio {wavs[0]} for {speaker}")

        # Convert script
        traditional = converter.convert(simplified)
        pinyin_str = get_pinyin(simplified)

        print(f"[{speaker}] {simplified}")

        # Copy audio
        shutil.copy2(expected_wav, audio_dest)

        # Extract pitch
        pitch_path = os.path.join(PITCH_DIR, f"{sentence_id}_pitch.json")
        pitch_values = extract_pitch(audio_dest)
        with open(pitch_path, "w") as f:
            json.dump(pitch_values, f)

        # Aeneas alignment (per character, skip punctuation)
        chars = [c for c in simplified if c.strip() and not c in "，。！？、""''（）《》【】"]
        align_path = os.path.join(ALIGN_DIR, f"{sentence_id}_align.json")
        try:
            run_aeneas(audio_dest, chars, align_path)
        except Exception as e:
            print(f"[error] aeneas failed for {speaker}: {e}")
            continue

        # Build aligned_pitch.json
        aligned = build_aligned_pitch(pitch_values, align_path, audio_dest)
        out_path = os.path.join(ALIGNED_PITCH_DIR, f"{sentence_id}_aligned_pitch.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f, indent=2)

        # Update sentences.json
        entry = {
            "audio_filename": f"{sentence_id}.wav",
            "chinese_simplified": simplified,
            "chinese_traditional": traditional,
            "english": "",
            "pinyin": pinyin_str,
        }
        sentences = [s for s in sentences if s["audio_filename"] != entry["audio_filename"]]
        sentences.append(entry)
        save_sentences_json(sentences)

        # Update DB
        upsert_db(sentence_id, simplified, traditional, "", pinyin_str, f"{sentence_id}.wav")

        print(f"  -> done: {sentence_id}")

    print("\nAll done.")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
