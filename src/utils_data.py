import random
from typing import List, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def load_data(path: str, debug: bool = False) -> Tuple[List[str], List[str]]:
    with open(path, "r") as f:
        data = f.read()

    words = data.splitlines()
    words = [word.strip() for word in words]  # Remove leading/trailing whitespace
    words = [word for word in words if word]  # Remove empty strings

    vocab = sorted(list(set("".join(words))))
    vocab = ["<eos>"] + vocab
    if debug:
        print(f"number of examples in dataset: {len(words)}")
        print(f"max word length: {max([len(word) for word in words])}")
        print(f"min word length: {min([len(word) for word in words])}")
        print(f"unique characters in dataset: {len(vocab)}")
        print("vocabulary:")
        print(" ".join(vocab))
        print("example for a word:")
    print(words[0])
    return words, vocab


def encode(word: str, vocab: List[str]) -> List[int]:
    """
    Encode a word, add <eos> at the beginning and the end of the word.
    """
    return [vocab.index("<eos>")] + [vocab.index(char) for char in word] + [vocab.index("<eos>")]


def decode(indices: List[int], vocab: List[str]) -> str:
    """
    Decode a list of indices to a word using the vocabulary.
    """
    return "".join([vocab[index] for index in indices])


def get_dataset(encoded_words: List[List[int]], block_size: int) -> Tuple[Array, Array]:
    """
    Take block size letters to predict the next letter.
    """
    X = []
    y = []
    for word in encoded_words:
        context = [0] * block_size
        for token in word[1:]:
            X.append(context)
            y.append(token)
            context = context[1:] + [token]
    return jnp.array(X), jnp.array(y)


def get_train_val_test(
    encoded_words: List[List[int]], block_size: int
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Split the dataset into training, validation and test sets.
    """
    random.shuffle(encoded_words)
    train_words = encoded_words[: int(0.8 * len(encoded_words))]
    val_words = encoded_words[int(0.8 * len(encoded_words)) : int(0.9 * len(encoded_words))]
    test_words = encoded_words[int(0.9 * len(encoded_words)) :]
    X_train, y_train = get_dataset(train_words, block_size)
    X_val, y_val = get_dataset(val_words, block_size)
    X_test, y_test = get_dataset(test_words, block_size)
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    words, vocab = load_data("/home/simon/mlp_language_modelling_jax/data/names.txt", debug=True)
    encoded_words = [encode(word, vocab) for word in words]
    X_tr, y_tr, X_val, y_val, X_test, y_test = get_train_val_test(encoded_words, block_size=3)
    print("Print some examples from the training set:")
    for i in range(16):
        print(f"{decode(X_tr[i].tolist(), vocab)} -> {decode([y_tr[i].item()], vocab)}")
