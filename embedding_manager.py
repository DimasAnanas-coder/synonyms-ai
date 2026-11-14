from pathlib import Path
import random
from typing import Optional

from matrix import Matrix


class EmbeddingManager:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = directory or Path(__file__).parent / "embeddings"

    def randomize(self, rows: int, cols: int) -> Matrix:
        """Create a matrix with random embedding values from 0.01 to 1.0."""
        data = [
            [random.randint(1, 100) / 100 for _ in range(cols)]
            for _ in range(rows)
        ]
        return Matrix(rows, cols, data)

    def save(self, embeddings: Matrix, name: str, epoch: int) -> Path:
        """Save a matrix as comma-separated rows in the embeddings directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{name}_{epoch}.txt"
        with path.open("w", encoding="utf-8") as output_file:
            for row in embeddings.data:
                output_file.write(",".join(map(str, row)) + "\n")
        return path

    def load(self, name: str, epoch: int) -> Matrix:
        """Load a previously saved matrix from the embeddings directory."""
        path = self.directory / f"{name}_{epoch}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run the training first to create it."
            )

        with path.open("r", encoding="utf-8") as input_file:
            data = [
                [float(value) for value in line.strip().split(",")]
                for line in input_file
                if line.strip()
            ]

        if not data:
            raise ValueError(f"Embedding file is empty: {path}")

        column_count = len(data[0])
        if any(len(row) != column_count for row in data):
            raise ValueError(f"Embedding rows have different lengths: {path}")

        return Matrix(len(data), column_count, data)


embedding_manager = EmbeddingManager()
