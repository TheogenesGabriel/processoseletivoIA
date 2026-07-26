import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
# ---------------------------------------------------------------------------

def main():
    # Carrega o dataset MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizar as imagens para [0, 1] e ajustar shape para (28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    # Constroi a CNN
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Bloco 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloco 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloco 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # EarlyStopping monitorando a perda de validação
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    # Treinamento com split de validação
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Avaliação final no conjunto de teste
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    val_acc_final = history.history["val_accuracy"][-1]

    print(f"\nAcuracia de validacao final: {val_acc_final:.4f}")
    print(f"Acuracia no conjunto de teste: {test_acc:.4f}")

    # Salva o modelo treinado
    model.save("model.h5")
    print("\nModelo salvo como model.h5")


if __name__ == "__main__":
    main()