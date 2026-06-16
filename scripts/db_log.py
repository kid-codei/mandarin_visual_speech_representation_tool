import os
import json
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "attempts.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs("data", exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id   TEXT    NOT NULL,
                timestamp     TEXT    NOT NULL,
                mode          TEXT    NOT NULL,
                overall_score REAL    NOT NULL,
                segments_json TEXT    NOT NULL
            )
        """)
        conn.commit()


def log_attempt(sentence_id: str, mode: str, overall_score: float, segments: list):
    init_db()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO attempts (sentence_id, timestamp, mode, overall_score, segments_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                sentence_id,
                datetime.utcnow().isoformat(),
                mode,
                overall_score,
                json.dumps(segments),
            ),
        )
        conn.commit()


def get_history(sentence_id: str, limit: int = 20) -> list:
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, mode, overall_score, segments_json
               FROM attempts
               WHERE sentence_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (sentence_id, limit),
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "mode": row["mode"],
                "overall_score": row["overall_score"],
                "segments": json.loads(row["segments_json"]),
            }
        )
    return result
