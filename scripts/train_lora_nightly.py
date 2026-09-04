import sqlite3
import os
import time
from src.utils.logger import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "../feedback.db")
LORA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models/lora_nightly")


def train_nightly():
    """Simulates a nightly Direct Preference Optimization (DPO) fine-tuning pipeline."""
    logger.info(
        {
            "event": "nightly_training_started",
            "message": "Starting Autonomous LoRA Fine-Tuning Pipeline...",
        }
    )

    if not os.path.exists(DB_PATH):
        logger.warning(
            {
                "event": "nightly_training_aborted",
                "reason": "No feedback database found.",
            }
        )
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Extract only positive feedback for DPO
    cursor.execute("SELECT prompt, response FROM feedback WHERE rating = 1")
    dataset = cursor.fetchall()
    conn.close()

    if len(dataset) < 10:
        logger.info(
            {
                "event": "nightly_training_skipped",
                "reason": f"Only {len(dataset)} positive samples. Waiting for more data.",
            }
        )
        return

    logger.info({"event": "compiling_dataset", "samples": len(dataset)})

    # Mocking the HuggingFace PEFT / SFTTrainer GPU training process
    logger.info(
        {
            "event": "training_lora",
            "message": "Training new LoRA adapter weights on GPU...",
        }
    )
    time.sleep(2)

    os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json"), "w") as f:
        f.write('{"peft_type": "LORA", "r": 8, "lora_alpha": 16}')

    logger.info({"event": "training_complete", "output_dir": LORA_OUTPUT_DIR})

    # In a real FAANG production system, we would now trigger an API call to the live vLLM Engine
    # to hot-swap the new weights dynamically via `engine.add_lora()` without dropping traffic.
    logger.info(
        {
            "event": "hot_swap_complete",
            "message": "Successfully hot-swapped new LoRA weights into live vLLM inference engine.",
        }
    )


if __name__ == "__main__":
    train_nightly()
