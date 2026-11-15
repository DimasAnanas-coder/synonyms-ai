import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value (true/false)")


EMBEDDINGS_COUNT = int(os.getenv("EMBEDDINGS_COUNT", "10"))
EPOCH_COUNT = int(os.getenv("EPOCH_COUNT", "100"))
SYNONYM_WORD = os.getenv("SYNONYM_WORD", "война")
SYNONYM_EMBEDDINGS_NAME = os.getenv("SYNONYM_EMBEDDINGS_NAME", "embeddings")
SYNONYM_EPOCH = int(os.getenv("SYNONYM_EPOCH", "1"))
TRAIN_FROM_SCRATCH = _get_bool("TRAIN_FROM_SCRATCH", True)
CONTINUE_INPUT_EMBEDDINGS = os.getenv("CONTINUE_INPUT_EMBEDDINGS", "").strip()
CONTINUE_OUTPUT_EMBEDDINGS = os.getenv("CONTINUE_OUTPUT_EMBEDDINGS", "").strip()
CONTINUE_OUTPUT_NAME = os.getenv("CONTINUE_OUTPUT_NAME", "embeddings_finetuned")
