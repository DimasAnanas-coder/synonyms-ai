from functools import lru_cache
from typing import Optional

import pymorphy3


class FilterManager:
    STOP_WORDS = {
        "а", "без", "более", "бы", "быть", "в", "во", "вот", "все", "весь", "вы",
        "где", "да", "даже", "для", "до", "его", "ее", "ей", "если", "есть",
        "еще", "ещё", "же", "за", "и", "из", "или", "им", "их", "к", "как", "ко",
        "когда", "кто", "ли", "либо", "мне", "можно", "мы", "на", "над",
        "надо", "наш", "не", "него", "нее", "нет", "ни", "них", "но", "ну",
        "о", "об", "однако", "он", "она", "они", "оно", "от", "перед", "по",
        "под", "при", "про", "с", "со", "так", "также", "там", "те", "тем",
        "то", "того", "тоже", "только", "том", "ты", "у", "уже", "хотя", "всё",
        "чего", "чем", "что", "чтобы", "эта", "эти", "это", "этот", "я",
    }

    def __init__(self):
        self._morph = pymorphy3.MorphAnalyzer(lang="ru")

    @lru_cache(maxsize=50_000)
    def _analyze(self, word: str) -> tuple[str, Optional[str]]:
        normalized_word = word.lower()
        parsed = self._morph.parse(normalized_word)[0]
        lemma = parsed.normal_form.lower()
        return lemma, parsed.tag.POS

    def normalize_word(self, word: str) -> str:
        return self._analyze(word)[0]

    def is_stop_word(self, word: str) -> bool:
        word = word.lower()
        return word in self.STOP_WORDS or self.normalize_word(word) in self.STOP_WORDS

    def filter_candidates(self, query: str, candidates, top_n: int):
        if top_n <= 0 or self.is_stop_word(query):
            return []

        query_lemma, query_pos = self._analyze(query)
        filtered = []
        for candidate, score in candidates:
            candidate_lemma, candidate_pos = self._analyze(candidate)
            if candidate_lemma == query_lemma or candidate_lemma in self.STOP_WORDS:
                continue
            if query_pos and candidate_pos and candidate_pos != query_pos:
                continue

            filtered.append((candidate_lemma, score))
            if len(filtered) == top_n:
                break

        return filtered


filter_manager = FilterManager()
