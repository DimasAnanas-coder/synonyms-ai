from pathlib import Path

from matrix import Matrix
from words import word_manager


def load_w_in():
    embeddings_directory = Path(__file__).parent / "embeddings"
    w_in_path = embeddings_directory / "embeddings_in_90.txt"
    if not w_in_path.exists():
        raise FileNotFoundError(f"{w_in_path} does not exist. Please run the training first.")
    
    with w_in_path.open("r", encoding="utf-8") as f:
        data = [list(map(float, line.strip().split(','))) for line in f]
    
    return Matrix(len(data), len(data[0]), data)

def find_synonyms(word: str, top_n: int = 5):
    w_in = load_w_in()
    if word not in word_manager.words:
        raise ValueError(f"Word '{word}' not found in the vocabulary.")
    
    word_id = word_manager.words[word]
    word_vector = w_in[word_id]

    similarities = []
    for other_word, other_id in word_manager.words.items():
        if other_word == word:
            continue
        other_vector = w_in[other_id]
        similarity = (word_vector.transpose() * other_vector) / (word_vector.norm() * other_vector.norm()) 
        similarities.append((other_word, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    word = "машина"
    top_n = 10
    synonyms = find_synonyms(word, top_n)
    for synonym, similarity in synonyms:
        print(f"{synonym}: {similarity}")