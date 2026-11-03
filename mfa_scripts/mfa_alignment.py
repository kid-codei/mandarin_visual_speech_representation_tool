"""
MFA alignment script for the Mandarin Visual Speech Representation Tool.

Run from the project root:
    python mfa_scripts/mfa_alignment.py

Requires MFA to be installed:
    conda install -c conda-forge montreal-forced-aligner
"""
import os
import shutil
import sqlite3
import subprocess
import sys

# Resolve project root regardless of where the script is invoked from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(PROJECT_ROOT, "db", "sentences.db")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "static", "audio")
TEXTGRID_DIR = os.path.join(PROJECT_ROOT, "static", "textgrid")
MFA_CORPUS_DIR = os.path.join(PROJECT_ROOT, "mfa_corpus")

MFA_DICTIONARY = os.path.join(PROJECT_ROOT, "mfa_resources", "mandarin_pinyin.dict")
MFA_MODEL = os.path.join(PROJECT_ROOT, "mfa_resources", "mandarin_mfa.zip")


def ensure_dirs():
    os.makedirs(TEXTGRID_DIR, exist_ok=True)
    if os.path.exists(MFA_CORPUS_DIR):
        shutil.rmtree(MFA_CORPUS_DIR)
    os.makedirs(MFA_CORPUS_DIR, exist_ok=True)


def load_sentences_from_db():
    """Return [(audio_filename, pinyin), ...] from the database."""
    if not os.path.exists(DB_PATH):
        print(f"[error] Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
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
    """Create .wav + .lab pairs in MFA_CORPUS_DIR."""
    for audio_filename, transcript in rows:
        audio_src = os.path.join(AUDIO_DIR, audio_filename)
        if not os.path.exists(audio_src):
            print(f"[skip] Missing audio file: {audio_src}")
            continue

        base = os.path.splitext(audio_filename)[0]
        audio_dst = os.path.join(MFA_CORPUS_DIR, f"{base}.wav")
        lab_dst = os.path.join(MFA_CORPUS_DIR, f"{base}.lab")

        shutil.copy2(audio_src, audio_dst)
        with open(lab_dst, "w", encoding="utf-8") as f:
            f.write(transcript.strip())

        print(f"[corpus] Prepared {base}.wav + {base}.lab")


def run_mfa_align():
    """Run MFA 'align' and move resulting TextGrids to TEXTGRID_DIR."""
    mfa_output = os.path.join(PROJECT_ROOT, "mfa_output")
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[MFA] stderr:", result.stderr)
        raise RuntimeError(f"MFA exited with code {result.returncode}")

    print("[MFA] Alignment complete.")

    for fname in os.listdir(mfa_output):
        if fname.endswith(".TextGrid"):
            src = os.path.join(mfa_output, fname)
            dst = os.path.join(TEXTGRID_DIR, fname)
            shutil.move(src, dst)
            print(f"[TextGrid] Saved {dst}")

    shutil.rmtree(mfa_output)


def main():
    ensure_dirs()
    rows = load_sentences_from_db()
    if not rows:
        print("[error] No sentences found in DB.")
        sys.exit(1)

    print(f"[DB] Loaded {len(rows)} sentences.")
    prepare_corpus_files(rows)
    run_mfa_align()
    print(f"Done. TextGrids saved to: {TEXTGRID_DIR}")


if __name__ == "__main__":
    main()
