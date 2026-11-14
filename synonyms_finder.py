from embedding_manager import embedding_manager
from words import word_manager


def find_synonyms(word: str, top_n: int = 5):
    w_in = embedding_manager.load("embeddings_in", 90)
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
