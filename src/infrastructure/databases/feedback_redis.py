import sqlite3
import os
from datetime import datetime
from src.utils.logger import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "../../feedback.db")


def init_db():
    """Initialize the SQLite database to store fine-tuning data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # This schema matches the required format for Direct Preference Optimization (DPO)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                rating INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error({"event": "feedback_db_init_failed", "error": str(e)})


def save_feedback(request_id: str, prompt: str, response: str, rating: int):
    """
    Saves user feedback for RLHF / DPO fine-tuning.
    Rating should be 1 (thumbs up) or -1 (thumbs down).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (request_id, prompt, response, rating, timestamp) VALUES (?, ?, ?, ?, ?)",
            (request_id, prompt, response, rating, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info(
            {"event": "feedback_saved", "request_id": request_id, "rating": rating}
        )
    except Exception as e:
        logger.error({"event": "feedback_save_failed", "error": str(e)})


# Initialize on import
init_db()
