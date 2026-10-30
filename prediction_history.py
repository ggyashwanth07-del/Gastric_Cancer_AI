
import json
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "prediction_history.db"


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                username TEXT,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                probabilities TEXT,
                image_width INTEGER,
                image_height INTEGER,
                quality_status TEXT
            )
        """)


def save_prediction(
    username,
    filename,
    predicted_class,
    confidence,
    probabilities,
    width,
    height,
    quality_status,
):
    init_db()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                username, filename, predicted_class, confidence,
                probabilities, image_width, image_height, quality_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            filename,
            predicted_class,
            float(confidence),
            json.dumps(probabilities),
            int(width),
            int(height),
            quality_status,
        ))


def get_history(limit=500):
    init_db()

    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT id, timestamp, username, filename,
                   predicted_class, confidence,
                   image_width, image_height, quality_status
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )
