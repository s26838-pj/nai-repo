"""
Projekt: Rozpoznawanie cyfr pisanych ręcznie (MNIST)
Autorzy: Szymon Stefański, Robert Elwart

Opis:
    Program realizuje zadanie klasyfikacji cyfr pisanych ręcznie
    z wykorzystaniem sztucznej sieci neuronowej.
    Wykorzystano klasyczny zbiór danych MNIST,
    który zawiera obrazy cyfr od 0 do 9.

    Jest to prosty i czytelny przykład własnego przypadku użycia
    sieci neuronowych do problemu klasyfikacji.

Cele projektu:
    - nauczenie sieci neuronowej rozpoznawania cyfr,
    - zapoznanie się z klasyfikacją obrazów w skali szarości,
    - realizacja zadania nr 4 (własny przypadek użycia).

Instrukcja uruchomienia:
    1. Zainstaluj Python 3.9 lub nowszy.
    2. W IDE (np. PyCharm, VS Code) ustaw interpreter Pythona.
    3. W terminalu zainstaluj wymagane biblioteki: pip install tensorflow
    4. Uruchom program poleceniem: python mnist.py
    Lub wciskając ręcznie przycisk Run w IDE.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def load_mnist():
    """
    Opis:
        Wczytuje zbiór danych MNIST z biblioteki TensorFlow.

    Zwraca:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    return tf.keras.datasets.mnist.load_data()


def preprocess_images(X_train, X_test):
    """
    Opis:
        Normalizuje obrazy do zakresu [0, 1]
        oraz spłaszcza je do postaci wektorów.

    Parametry:
        X_train: Obrazy treningowe.
        X_test: Obrazy testowe.

    Zwraca:
        tuple: Przetworzone X_train, X_test
    """
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    return X_train, X_test


def create_model():
    """
    Opis:
        Tworzy prostą sieć neuronową typu MLP
        do klasyfikacji cyfr MNIST.

    Zwraca:
        tf.keras.Model: Skonstruowany model sieci neuronowej.
    """
    model = models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(784,)))
    model.add(layers.Dense(64, activation="relu"))
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
        Trenuje sieć neuronową na zbiorze MNIST.

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
        epochs=5,
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
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_train, X_test = preprocess_images(X_train, X_test)

    model = create_model()

    print("Trenowanie sieci neuronowej dla MNIST")
    train_model(model, X_train, y_train, X_test, y_test)

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nSkuteczność sieci neuronowej: {accuracy:.4f}")


if __name__ == "__main__":
    main()
