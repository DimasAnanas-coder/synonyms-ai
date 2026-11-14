from collections.abc import Iterator
from pathlib import Path


BOOKS_DIR = Path(__file__).parent / "books"


def load_text() -> Iterator[str]:
    for book_path in sorted(BOOKS_DIR.glob("*.txt")):
        with book_path.open(encoding="utf-8") as book_file:
            yield from book_file
