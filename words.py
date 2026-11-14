from collections import deque
from collections.abc import Callable, Iterator

from filter_manager import filter_manager
from texts.text import load_text


class WordManager:
    def __init__(self, lines_factory: Callable[[], Iterator[str]]):
        self.lines_factory = lines_factory
        self._words = self._build_words_dict()

    def _word_iterator(self) -> Iterator[str]:
        for line in self.lines_factory():
            word = ""
            for symbol in line:
                if symbol.isalpha():
                    word += symbol
                elif word:
                    yield filter_manager.normalize_word(word)
                    word = ""
            if word:
                yield filter_manager.normalize_word(word)

    def _build_words_dict(self) -> dict[str, int]:
        words: dict[str, int] = {}
        for word in self._word_iterator():
            if word not in words:
                words[word] = len(words)
        return words

    @property
    def words(self) -> dict[str, int]:
        return self._words

    def iter_word_ids(self) -> Iterator[int]:
        for word in self._word_iterator():
            yield self._words[word]

    def iter_context_pairs(self, window_size: int = 2) -> Iterator[tuple[int, int]]:
        if window_size < 0:
            raise ValueError("window_size must be non-negative")

        token_ids = iter(self.iter_word_ids())
        try:
            center_id = next(token_ids)
        except StopIteration:
            return

        future = deque()
        for _ in range(window_size):
            try:
                future.append(next(token_ids))
            except StopIteration:
                break

        past = deque(maxlen=window_size)
        while True:
            for context_id in (*past, *future):
                yield center_id, context_id

            past.append(center_id)
            if not future:
                return
            center_id = future.popleft()

            try:
                future.append(next(token_ids))
            except StopIteration:
                pass


word_manager = WordManager(load_text)
