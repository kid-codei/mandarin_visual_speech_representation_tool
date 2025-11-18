import os
import shutil
import sqlite3
import subprocess

# ---------- CONFIGURE THESE PATHS ----------
DB_PATH = "db/sentences.db"          # adjust if different
AUDIO_DIR = "static/audio"           # where your .wav files live
TEXTGRID_DIR = "static/textgrid"     # where we want MFA outputs

# Temporary MFA corpus directory
MFA_CORPUS_DIR = "mfa_corpus"

# Paths to MFA resources - YOU MUST ADJUST THESE
MFA_DICTIONARY = "/Users/johnhenrycruz/mandarin_visual_speech_representation_tool/mfa_resources/mandarin_pinyin.dict" # e.g. pinyin lexicon
MFA_MODEL = "/Users/johnhenrycruz/mandarin_visual_speech_representation_tool/mfa_resources/mandarin_mfa.zip"  # MFA acoustic model
# -------------------------------------------


def ensure_dirs():
    os.makedirs(TEXTGRID_DIR, exist_ok=True)
    if os.path.exists(MFA_CORPUS_DIR):
        shutil.rmtree(MFA_CORPUS_DIR)
    os.makedirs(MFA_CORPUS_DIR, exist_ok=True)


def load_sentences_from_db():
    """
    Returns a list of (audio_filename, transcript_str).

    Assumes a table 'sentences' with columns:
      - audio_filename (e.g., 'f1.wav')
      - pinyin OR traditional/simplified (whichever you want MFA to see)

    Tweak the SELECT and transcript field for your actual schema.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # EXAMPLE: using pinyin as transcript
    # If your column is different, change 'pinyin' below.
    cur.execute("""
        SELECT audio_filename, pinyin
        FROM sentences
        WHERE audio_filename IS NOT NULL
          AND pinyin IS NOT NULL
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


def prepare_corpus_files(rows):
    """
    Given rows [(audio_filename, transcript), ...],
    create .wav + .lab pairs in MFA_CORPUS_DIR.
    """
    for audio_filename, transcript in rows:
        audio_src = os.path.join(AUDIO_DIR, audio_filename)
        if not os.path.exists(audio_src):
            print(f"[skip] Missing audio file: {audio_src}")
            continue

        base = os.path.splitext(audio_filename)[0]
        audio_dst = os.path.join(MFA_CORPUS_DIR, f"{base}.wav")
        lab_dst = os.path.join(MFA_CORPUS_DIR, f"{base}.lab")

        # Copy or symlink the audio (copy is simpler & more portable)
        shutil.copy2(audio_src, audio_dst)

        # Write transcript to .lab
        with open(lab_dst, "w", encoding="utf-8") as f:
            f.write(transcript.strip())

        print(f"[corpus] Prepared {audio_dst} and {lab_dst}")


def run_mfa_align():
    """
    Run MFA 'align' over the corpus.
    Produces TextGrids in an output folder, which we then move to TEXTGRID_DIR.
    """
    mfa_output = "mfa_output"
    if os.path.exists(mfa_output):
        shutil.rmtree(mfa_output)

    os.makedirs(mfa_output, exist_ok=True)

    cmd = [
        "mfa", "align",
        MFA_CORPUS_DIR,
        MFA_DICTIONARY,
        MFA_MODEL,
        mfa_output,
        "--clean",
    ]

    print("[MFA] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[MFA] Alignment complete.")

    # Move all TextGrids to TEXTGRID_DIR
    for fname in os.listdir(mfa_output):
        if fname.endswith(".TextGrid"):
            src = os.path.join(mfa_output, fname)
            dst = os.path.join(TEXTGRID_DIR, fname)
            shutil.move(src, dst)
            print(f"[TextGrid] {dst}")

    # Optional cleanup
    shutil.rmtree(mfa_output)


def main():
    ensure_dirs()
    rows = load_sentences_from_db()
    if not rows:
        print("No sentences found in DB. Check your SELECT query.")
        return

    print(f"[DB] Loaded {len(rows)} sentences.")
    prepare_corpus_files(rows)
    run_mfa_align()
    print(f"Done. TextGrids saved to: {TEXTGRID_DIR}")


if __name__ == "__main__":
    main()
