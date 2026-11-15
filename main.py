from math import exp

from embedding_manager import embedding_manager
from matrix import Matrix, Vector

from words import word_manager


words_count = len(word_manager.words)
embeddings_count = 10


def calculate_scalar_product(vec1: Matrix, vec2: Matrix):
    return vec1.transpose() * vec2


epoch_count = 100
study_rate = 0.01

w_in = embedding_manager.randomize(words_count, embeddings_count)
w_out = embedding_manager.randomize(words_count, embeddings_count)


def epoch_trachiing_event(epoch_number, w_in_now, w_out_now, filename="embeddings"):
    print(f"Epoch {epoch_number} completed.")
    embedding_manager.save(w_in_now, f"{filename}_in", epoch_number)
    embedding_manager.save(w_out_now, f"{filename}_out", epoch_number)

def _train(w_in_now, w_out_now, epochs, filename, start_epoch=0):
    for epoch in range(start_epoch, start_epoch + epochs):
        for i, j in word_manager.iter_context_pairs():
            #print(f"Processing pair: ({i}, {j})")  # Debugging line
            vi: Vector = w_in_now[i]
            z = Vector(words_count)

            uj: Vector = w_out_now[j]

            for k in word_manager.words.values():
                z[k] = calculate_scalar_product(vi, w_out_now[k])

            sum_exp = sum(exp(zk) for zk in z)
            p = Vector(words_count, [exp(zk) / sum_exp for zk in z])

            grad_vi = -uj + w_out_now.transpose() * p
            w_in_now[i] = vi - grad_vi * study_rate

            for k in word_manager.words.values():
                grad_uj = vi * p[k]
                if k == j:
                    grad_uj -= vi

                w_out_now[k] = w_out_now[k] - grad_uj * study_rate
        epoch_trachiing_event(epoch, w_in_now, w_out_now, filename)


def train_embeddings():
    """Train a new model from random embeddings."""
    _train(w_in, w_out, epoch_count, "embeddings")


def continue_training(input_embeddings_path, output_embeddings_path, epochs,
                      filename="embeddings_finetuned", start_epoch=0):
    """Continue training from saved input and output embedding matrix files."""
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    w_in_now = embedding_manager.load_file(input_embeddings_path)
    w_out_now = embedding_manager.load_file(output_embeddings_path)
    if w_in_now.rows != words_count:
        raise ValueError("Input embeddings must have one row per current vocabulary word")
    if (w_out_now.rows, w_out_now.cols) != (w_in_now.rows, w_in_now.cols):
        raise ValueError("Output embeddings must match the input embedding dimensions")
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative")

    _train(w_in_now, w_out_now, epochs, filename, start_epoch)
    return w_in_now, w_out_now

if __name__ == "__main__":
    train_embeddings()        

