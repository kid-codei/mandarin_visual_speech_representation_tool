from pypinyin import pinyin, Style
from opencc import OpenCC
import sqlite3

# will convert traditional characters to simplified
converter = OpenCC('t2s')

# get pinyin from the character
def get_pinyin(text):
    return ' '.join([syllable[0] for syllable in pinyin(text, style=Style.TONE3)])

# use cursor to get rows from 
conn = sqlite3.connect("db/sentences.db")
cursor = conn.cursor()
cursor.execute("SELECT id, chinese_traditional FROM Sentences")
rows = cursor.fetchall()

# get simplified and pinyin text for traditional sentence
# update the database
for id_, trad_text in rows:
    simp_text = converter.convert(trad_text)
    pinyin_text = get_pinyin(trad_text)
    cursor.execute("""
        UPDATE Sentences
        SET chinese_simplified = ?, pinyin = ?
        WHERE id = ?
    """, (simp_text, pinyin_text, id_))

conn.commit()
conn.close()