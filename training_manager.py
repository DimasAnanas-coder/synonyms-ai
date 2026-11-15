from math import exp

from embedding_manager import EmbeddingManager, embedding_manager
from matrix import Matrix, Vector
from words import WordManager, word_manager


class TrainingManager:
    def __init__(
        self,
        embeddings: EmbeddingManager = embedding_manager,
        words: WordManager = word_manager,
        study_rate: float = 0.01,
    ):
        self.embeddings = embeddings
        self.words = words
        self.study_rate = study_rate

    def _train(self, w_in: Matrix, w_out: Matrix, epochs: int,
               filename: str, start_epoch: int = 0) -> None:
        words_count = len(self.words.words)
        self._validate_matrices(w_in, w_out, words_count)
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")

        for epoch in range(start_epoch, start_epoch + epochs):
            for i, j in self.words.iter_context_pairs():
                vi: Vector = w_in[i]
                z = Vector(words_count)
                uj: Vector = w_out[j]

                for k in self.words.words.values():
                    z[k] = vi.transpose() * w_out[k]

                sum_exp = sum(exp(value) for value in z)
                probabilities = Vector(
                    words_count,
                    [exp(value) / sum_exp for value in z],
                )

                grad_vi = -uj + w_out.transpose() * probabilities
                w_in[i] = vi - grad_vi * self.study_rate

                for k in self.words.words.values():
                    grad_uj = vi * probabilities[k]
                    if k == j:
                        grad_uj -= vi
                    w_out[k] = w_out[k] - grad_uj * self.study_rate

            self._save_epoch(w_in, w_out, filename, epoch)

    @staticmethod
    def _validate_matrices(w_in: Matrix, w_out: Matrix, words_count: int) -> None:
        if w_in.rows != words_count:
            raise ValueError("Input embeddings must have one row per current vocabulary word")
        if (w_out.rows, w_out.cols) != (w_in.rows, w_in.cols):
            raise ValueError("Output embeddings must match the input embedding dimensions")

    def _save_epoch(self, w_in: Matrix, w_out: Matrix,
                    filename: str, epoch: int) -> None:
        print(f"Epoch {epoch} completed.")
        self.embeddings.save(w_in, f"{filename}_in", epoch)
        self.embeddings.save(w_out, f"{filename}_out", epoch)

    def train_embeddings(self, w_in: Matrix, w_out: Matrix, epochs: int = 100,
                         filename: str = "embeddings") -> tuple[Matrix, Matrix]:
        """Train from the matrices provided by the caller."""
        self._train(w_in, w_out, epochs, filename)
        return w_in, w_out

    def continue_training(
        self,
        input_embeddings_path: str,
        output_embeddings_path: str,
        epochs: int,
        filename: str = "embeddings_finetuned",
        start_epoch: int = 0,
    ) -> tuple[Matrix, Matrix]:
        """Continue training from saved input and output embedding files."""
        w_in = self.embeddings.load_file(input_embeddings_path)
        w_out = self.embeddings.load_file(output_embeddings_path)
        self._train(w_in, w_out, epochs, filename, start_epoch)
        return w_in, w_out


training_manager = TrainingManager()
