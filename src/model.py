from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

import utils_data

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
def train_step(params: MLPParams, X: Array, y: Array, learning_rate: float) -> MLPParams:
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
) -> MLPParams:
    for epoch in range(num_epochs):
        ix = random.randint(random.PRNGKey(epoch), (batch_size,), 0, len(X_tr))
        X_batch = jnp.take(X_tr, ix, axis=0)
        y_batch = jnp.take(y_tr, ix, axis=0)
        params, loss = train_step(params, X_batch, y_batch, learning_rate)
        if epoch % 1000 == 0:
            val_loss = evaluate(params, X_val, y_val)
            print(f"epoch {epoch}, loss {loss}, val_loss {val_loss}")
    return params, loss


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
    params, loss = train(
        params,
        X_tr,
        y_tr,
        X_val,
        y_val,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
