from embedding_manager import embedding_manager
from training_manager import training_manager
from words import word_manager


EMBEDDINGS_COUNT = 10
EPOCH_COUNT = 100


def main():
    words_count = len(word_manager.words)
    w_in = embedding_manager.randomize(words_count, EMBEDDINGS_COUNT)
    w_out = embedding_manager.randomize(words_count, EMBEDDINGS_COUNT)
    training_manager.train_embeddings(w_in, w_out, epochs=EPOCH_COUNT)


if __name__ == "__main__":
    main()
