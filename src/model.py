from typing import Any, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

import src.utils_data as utils_data

Array = jax.Array


class MLPParams(NamedTuple):
    embedding: Array
    W1: Array
    b1: Array
    W2: Array
    b2: Array


def init_params(
    batch_size: int, vocab_size: int, block_size: int, embed_size: int, hidden_size: int
) -> MLPParams:
    keys = random.split(random.PRNGKey(42), 5)
    embedding = random.normal(keys[0], (vocab_size, embed_size))
    W1 = random.normal(keys[1], (block_size * embed_size, hidden_size))
    b1 = random.normal(keys[2], (hidden_size,))
    W2 = random.normal(keys[3], (hidden_size, vocab_size))
    b2 = random.normal(keys[4], (vocab_size,))
    return MLPParams(embedding, W1, b1, W2, b2)


def forward(params: MLPParams, X: Array) -> Array:
    embedded = params.embedding[X]  # (batch_size, block_size, embed_size)
    embedded = embedded.reshape(X.shape[0], -1)  # (batch_size, block_size * embed_size)
    hidden = jnp.tanh(embedded.dot(params.W1) + params.b1)  # (batch_size, hidden_size)
    output = hidden.dot(params.W2) + params.b2  # (batch_size, vocab_size)
    return output


def loss_fn(params: MLPParams, X: Array, y: Array) -> Array:
    logits = forward(params, X)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(y.size), y])


@jax.jit
def train_step(
    params: MLPParams, X: Array, y: Array, learning_rate: float
) -> Tuple[MLPParams, Array]:
    loss = loss_fn(params, X, y)
    grads = jax.grad(loss_fn)(params, X, y)
    params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


def evaluate(params: MLPParams, X: Array, y: Array) -> Array:
    logits = forward(params, X)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(y.size), y])


def train(
    params: MLPParams,
    X_tr: Array,
    y_tr: Array,
    X_val: Array,
    y_val: Array,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Tuple[MLPParams, Array, List[int], List[Any], List[Any]]:
    epochs = []
    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        ix = random.randint(random.PRNGKey(epoch), (batch_size,), 0, len(X_tr))
        X_batch = jnp.take(X_tr, ix, axis=0)
        y_batch = jnp.take(y_tr, ix, axis=0)
        params, loss = train_step(params, X_batch, y_batch, learning_rate)
        if epoch % 1000 == 0 and epoch > 0:
            val_loss = evaluate(params, X_val, y_val)
            print(f"epoch {epoch}, loss {loss}, val_loss {val_loss}")
            epochs.append(epoch)
            losses.append(loss.item())
            val_losses.append(val_loss.item())
    return params, loss, epochs, losses, val_losses


def sample(params: MLPParams, key: Array, vocab: List[str]) -> str:
    """
    1) Start with <eos>
    2) Index into the weights matrix W for the current character
    3) Sample the next character from the distribution
    4) Append the sampled character to the sampled word
    5) Repeat steps 3-5 until <eos> is sampled
    6) Return the sampled word
    """
    current_chars = jnp.array([vocab.index("<eos>"), vocab.index("<eos>"), vocab.index("<eos>")])[
        None, :
    ]
    sampled_word = ["<eos>", "<eos>", "<eos>"]
    while True:
        key, subkey = jax.random.split(key)
        logits = forward(params, current_chars)
        sampled_char = random.categorical(subkey, logits=logits)[0]
        current_chars = jnp.concatenate(
            [current_chars[:, 1:], jnp.array([sampled_char])[None, :]], axis=1
        )
        sampled_word.append(vocab[sampled_char])
        if sampled_char == vocab.index("<eos>"):
            break
    return "".join(sampled_word)[len("<eos><eos><eos>") : -len("<eos>")]


if __name__ == "__main__":
    batch_size = 64
    vocab_size = 27
    block_size = 3
    embed_size = 32
    hidden_size = 128
    learning_rate = 0.1
    num_epochs = 100_000
    params = init_params(batch_size, vocab_size, block_size, embed_size, hidden_size)
    words, vocab = utils_data.load_data("/home/simon/mlp_language_modelling_jax/data/names.txt")
    encoded_words = [utils_data.encode(word, vocab) for word in words]
    X_tr, y_tr, X_val, y_val, X_test, y_test = utils_data.get_train_val_test(
        encoded_words, block_size=block_size
    )
    params, loss, epochs, losses, val_losses = train(
        params,
        X_tr,
        y_tr,
        X_val,
        y_val,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    plt.figure()
    plt.plot(epochs, losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.legend()
    plt.savefig("losses.png")
    for i in range(10):
        print(sample(params, random.PRNGKey(i), vocab))
