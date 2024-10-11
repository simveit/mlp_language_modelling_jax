import matplotlib.pyplot as plt
from jax import random

from src.model import init_params, sample, train
from src.utils_data import encode, get_train_val_test, load_data

if __name__ == "__main__":
    batch_size = 64
    vocab_size = 27
    block_size = 3
    embed_size = 32
    hidden_size = 128
    learning_rate = 0.1
    num_epochs = 100_000
    params = init_params(batch_size, vocab_size, block_size, embed_size, hidden_size)
    words, vocab = load_data("/home/simon/mlp_language_modelling_jax/data/names.txt")
    encoded_words = [encode(word, vocab) for word in words]
    X_tr, y_tr, X_val, y_val, X_test, y_test = get_train_val_test(
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
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")

    min_val_loss = min(val_losses)

    plt.axhline(
        min_val_loss, color="r", linestyle="--", label=f"Min Validation Loss: {min_val_loss:.2f}"
    )

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.savefig("losses.png")

    for i in range(10):
        print(sample(params, random.PRNGKey(i), vocab))
