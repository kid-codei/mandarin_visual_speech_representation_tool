from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import sqlite3
import os

DB_PATH = "db/sentences.db"
AUDIO_DIR = "static/audio"
ALIGN_OUTPUT_DIR = "static/json/align"

# write func to get alignment from audio file
def run_alignment(audio_path, pinyin_text, output_json_path):
    # segment to less overwhelm tts
    lines = pinyin_text.split()
    lines_per_group = 2
    grouped_lines = [' '.join(lines[i:i+lines_per_group]) for i in range(0, len(lines), lines_per_group)]

    with open("temp_pinyin.txt", "w", encoding="utf-8") as f:
        f.write(pinyin_text)

    config_string = u"task_language=cmn|os_task_file_format=json|is_text_type=plain"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath("temp_pinyin.txt")
    task.sync_map_file_path_absolute = os.path.abspath(output_json_path)

    ExecuteTask(task).execute()
    task.output_sync_map_file()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT audio_filename, pinyin FROM Sentences")
rows = cursor.fetchall()

# iterate through all audios in db
# use run_alignment function
for audio_filename, pinyin_text in rows:
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    output_json_path = os.path.join(ALIGN_OUTPUT_DIR, audio_filename.replace(".wav", "_align.json"))

    # handle potneital execute errors
    if not os.path.exists(audio_path):
        print(f"Missing audio: {audio_path}")
        continue
    if not pinyin_text:
        print(F"No pinyin for {audio_filename}")
        continue

    run_alignment(audio_path, pinyin_text, output_json_path)

conn.close()