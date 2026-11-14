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

def train_embeddings():
    for epoch in range(epoch_count):
        for i, j in word_manager.iter_context_pairs():
            # print(f"Processing pair: ({i}, {j})")  # Debugging line
            vi: Vector = w_in[i]
            z = Vector(words_count)

            uj: Vector = w_out[j]

            for k in word_manager.words.values():
                z[k] = calculate_scalar_product(vi, w_out[k])

            sum_exp = sum(exp(zk) for zk in z)
            p = Vector(words_count, [exp(zk) / sum_exp for zk in z])

            grad_vi = -uj + w_out.transpose() * p
            w_in[i] = vi - grad_vi * study_rate

            for k in word_manager.words.values():
                grad_uj = vi * p[k]
                if k == j:
                    grad_uj -= vi

                w_out[k] = w_out[k] - grad_uj * study_rate
        if epoch % 10 == 0:
            epoch_trachiing_event(epoch, w_in, w_out)

if __name__ == "__main__":
    train_embeddings()        

