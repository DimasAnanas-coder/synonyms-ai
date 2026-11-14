import random
from math import exp
from pathlib import Path

from matrix import Matrix, Vector

from words import word_manager


words_count = len(word_manager.words)
embeddings_count = 10


def generate_embeddings() -> Matrix:
    def random_coord():
        return random.randint(1, 100) / 100
        
    data = [[random_coord() for _ in range(embeddings_count)] for _ in range(words_count)]
    return Matrix(words_count, embeddings_count, data)


def calculate_scalar_product(vec1: Matrix, vec2: Matrix):
    return vec1.transpose() * vec2


epoch_count = 100
study_rate = 0.01

w_in = generate_embeddings()
w_out = generate_embeddings()


def epoch_trachiing_event(epoch_number, w_in_now, w_out_now, filename="embeddings"):
    def save_embeddings_to_file(embeddings, filename):
        embeddings_directory = Path(__file__).parent / "embeddings"
        embeddings_directory.mkdir(parents=True, exist_ok=True)
        output_path = embeddings_directory / f"{filename}_{epoch_number}.txt"
        with output_path.open("w", encoding="utf-8") as f:
            for embedding in embeddings:
                f.write(','.join(map(str, embedding)) + '\n')

    print(f"Epoch {epoch_number} completed.")
    save_embeddings_to_file(w_in_now, f"{filename}_in")
    save_embeddings_to_file(w_out_now, f"{filename}_out")

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
                grad_uj = w_in[k] * p[k]
                if k == j:
                    grad_uj -= vi

                w_out[k] = w_out[k] - grad_uj * study_rate

        epoch_trachiing_event(epoch, w_in, w_out)

if __name__ == "__main__":
    train_embeddings()        

