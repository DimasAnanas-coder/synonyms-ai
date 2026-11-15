from embedding_manager import embedding_manager
from env import (
    CONTINUE_INPUT_EMBEDDINGS,
    CONTINUE_OUTPUT_NAME,
    CONTINUE_OUTPUT_EMBEDDINGS,
    EMBEDDINGS_COUNT,
    EPOCH_COUNT,
    TRAIN_FROM_SCRATCH,
)
from training_manager import training_manager
from words import word_manager


def main():
    if TRAIN_FROM_SCRATCH:
        words_count = len(word_manager.words)
        w_in = embedding_manager.randomize(words_count, EMBEDDINGS_COUNT)
        w_out = embedding_manager.randomize(words_count, EMBEDDINGS_COUNT)
        training_manager.train_embeddings(w_in, w_out, epochs=EPOCH_COUNT)
        return

    if not CONTINUE_INPUT_EMBEDDINGS or not CONTINUE_OUTPUT_EMBEDDINGS:
        raise ValueError(
            "Set CONTINUE_INPUT_EMBEDDINGS and CONTINUE_OUTPUT_EMBEDDINGS "
            "in .env when TRAIN_FROM_SCRATCH=false"
        )

    training_manager.continue_training(
        CONTINUE_INPUT_EMBEDDINGS,
        CONTINUE_OUTPUT_EMBEDDINGS,
        epochs=EPOCH_COUNT,
        filename=CONTINUE_OUTPUT_NAME,
    )


if __name__ == "__main__":
    main()
