"""
Projekt: Naucz sieć rozpoznać zwierzęta, np. z zbioru CIFAR10
Autorzy: Szymon Stefański, Robert Elwart

Opis:
    Program realizuje zadanie klasyfikacji obrazów zwierząt z wykorzystaniem
    konwolucyjnej sieci neuronowej (CNN). Wykorzystano zbiór CIFAR-10,
    zawierający kolorowe obrazy o rozmiarze 32x32 piksele.

    Zbiór CIFAR-10 składa się z 10 klas:
    samolot, samochód, ptak, kot, jeleń, pies, żaba, koń, statek, ciężarówka.

Cele projektu:
    - zapoznanie się z klasyfikacją obrazów przy użyciu CNN,
    - nauczenie sieci neuronowej rozpoznawania obiektów (zwierząt),
    - ocena skuteczności modelu na zbiorze testowym.

Instrukcja uruchomienia:
    1. Zainstaluj Python 3.9 lub nowszy.
    2. W IDE (np. PyCharm, VS Code) ustaw interpreter Pythona.
    3. W terminalu zainstaluj wymagane biblioteki: pip install tensorflow
    4. Uruchom program poleceniem: python cifar10.py
    Lub wciskając ręcznie przycisk Run w IDE.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


CLASSES = [
    "samolot", "samochód", "ptak", "kot", "jeleń",
    "pies", "żaba", "koń", "statek", "ciężarówka"
]


def load_cifar10():
    """
    Opis:
        Wczytuje zbiór danych CIFAR-10 z biblioteki TensorFlow.

    Zwraca:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    return tf.keras.datasets.cifar10.load_data()


def preprocess_images(X_train, X_test):
    """
    Opis:
        Normalizuje obrazy do zakresu [0, 1].

    Parametry:
        X_train: Obrazy treningowe.
        X_test: Obrazy testowe.

    Zwraca:
        tuple: X_train, X_test po normalizacji.
    """
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, X_test


def create_cnn_model():
    """
    Opis:
        Tworzy konwolucyjną sieć neuronową (CNN)
        do klasyfikacji obrazów CIFAR-10.

    Zwraca:
        tf.keras.Model: Skonstruowany model CNN.
    """
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
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
        Trenuje model CNN na zbiorze CIFAR-10.

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
        wczytanie danych, uczenie sieci oraz ewaluację.
    """
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    X_train, X_test = preprocess_images(X_train, X_test)

    model = create_cnn_model()

    print("Trenowanie sieci CNN dla CIFAR-10")
    train_model(model, X_train, y_train, X_test, y_test)

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nSkuteczność sieci neuronowej (CNN): {accuracy:.4f}")


if __name__ == "__main__":
    main()
