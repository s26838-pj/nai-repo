"""
Projekt: Rozpoznawanie ubrań z wykorzystaniem sieci neuronowych (Fashion-MNIST)
Autorzy: Szymon Stefański, Robert Elwart

Opis:
    Program realizuje zadanie klasyfikacji obrazów ubrań z wykorzystaniem
    sztucznej sieci neuronowej. Wykorzystano zbiór Fashion-MNIST,
    który jest zamiennikiem klasycznego MNIST i zawiera obrazy
    produktów odzieżowych.

    Zbiór Fashion-MNIST składa się z 10 klas:
    T-shirt/top, spodnie, sweter, sukienka, płaszcz,
    sandał, koszula, sneaker, torba, botek.

Cele projektu:
    - nauczenie sieci neuronowej rozpoznawania obrazów ubrań,
    - zapoznanie się z klasyfikacją obrazów w skali szarości,
    - porównanie zastosowania sieci neuronowych dla różnych typów danych.

Instrukcja uruchomienia:
    1. Zainstaluj Python 3.9 lub nowszy.
    2. W IDE (np. PyCharm, VS Code) ustaw interpreter Pythona.
    3. W terminalu zainstaluj wymagane biblioteki: pip install tensorflow
    4. Uruchom program poleceniem: python fashion_mnist.py
    Lub wciskając ręcznie przycisk Run w IDE.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


CLASSES = [
    "T-shirt/top", "Spodnie", "Sweter", "Sukienka", "Płaszcz",
    "Sandał", "Koszula", "Sneaker", "Torba", "Botek"
]


def load_fashion_mnist():
    """
    Opis:
        Wczytuje zbiór danych Fashion-MNIST z biblioteki TensorFlow.

    Zwraca:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    return tf.keras.datasets.fashion_mnist.load_data()


def preprocess_images(X_train, X_test):
    """
    Opis:
        Normalizuje obrazy do zakresu [0, 1]
        oraz dodaje wymiar kanału.

    Parametry:
        X_train: Obrazy treningowe.
        X_test: Obrazy testowe.

    Zwraca:
        tuple: Przetworzone X_train, X_test
    """
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]

    return X_train, X_test


def create_cnn_model():
    """
    Opis:
        Tworzy konwolucyjną sieć neuronową (CNN)
        do klasyfikacji obrazów Fashion-MNIST.

    Zwraca:
        tf.keras.Model: Skonstruowany model CNN.
    """
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Opis:
        Trenuje sieć neuronową na zbiorze Fashion-MNIST.

    Parametry:
        model: Model sieci neuronowej.
        X_train: Dane treningowe.
        y_train: Etykiety treningowe.
        X_test: Dane testowe.
        y_test: Etykiety testowe.

    Zwraca:
        History: Historia procesu uczenia.
    """
    return model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )


def evaluate_model(model, X_test, y_test):
    """
    Opis:
        Oblicza skuteczność (accuracy) modelu na zbiorze testowym.

    Parametry:
        model: Wytrenowany model.
        X_test: Dane testowe.
        y_test: Etykiety testowe.

    Zwraca:
        float: Skuteczność modelu.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


def main():
    """
    Opis:
        Funkcja główna programu realizująca pełny proces:
        wczytanie danych, trenowanie sieci oraz ewaluację.
    """
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    X_train, X_test = preprocess_images(X_train, X_test)

    model = create_cnn_model()

    print("Trenowanie sieci neuronowej dla Fashion-MNIST")
    train_model(model, X_train, y_train, X_test, y_test)

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nSkuteczność sieci neuronowej: {accuracy:.4f}")


if __name__ == "__main__":
    main()
