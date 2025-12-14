"""
Projekt: Klasyfikacja jakości wina z użyciem sieci neuronowych
Autorzy: Szymon Stefański, Robert Elwart

Opis:
    Program realizuje zadanie klasyfikacji jakości wina na podstawie
    cech fizykochemicznych z wykorzystaniem sztucznych sieci neuronowych
    (TensorFlow).

    Dataset: Wine Quality Dataset
    Źródło: UCI Machine Learning Repository

Cele projektu:
    - nauczenie sieci neuronowej rozwiązywać problem klasyfikacji,
    - porównanie dwóch architektur sieci neuronowych (mała i większa),
    - ocena jakości modelu przy użyciu confusion matrix.

Instrukcja uruchomienia:
    1. Zainstaluj Python 3.9 lub nowszy
    2. W IDE (np. PyCharm, VS Code) ustaw interpreter Pythona.
    3. W terminalu zainstaluj wymagane biblioteki: pip install tensorflow numpy pandas scikit-learn requests
    4. Uruchom program poleceniem: python wine_nn.py
    Lub wciskając ręcznie przycisk Run w IDE.
"""

import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)
DATASET_PATH = "winequality-red.csv"


def download_dataset(url: str, path: str) -> None:
    """
    Opis:
        Pobiera zbiór danych z internetu i zapisuje go lokalnie,
        jeśli plik nie istnieje w katalogu projektu.

    Parametry:
        url (str): Adres URL do pliku CSV.
        path (str): Lokalna ścieżka zapisu pliku.

    Zwraca:
        None
    """
    if os.path.exists(path):
        print("Dataset już istnieje – pomijam pobieranie.")
        return

    print("Pobieranie datasetu...")
    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as file:
        file.write(response.content)

    print("Dataset pobrany poprawnie.")


def load_data(path: str) -> pd.DataFrame:
    """
    Opis:
        Wczytuje zbiór danych Wine Quality z pliku CSV.

    Parametry:
        path (str): Ścieżka do pliku CSV.

    Zwraca:
        pd.DataFrame: Dane w postaci DataFrame.
    """
    return pd.read_csv(path, sep=";")


def process_data(df: pd.DataFrame):
    """
    Opis:
        Dzieli dane na cechy i etykiety, wykonuje standaryzację
        oraz podział na zbiór treningowy i testowy.

    Parametry:
        df (pd.DataFrame): Surowy zbiór danych.

    Zwraca:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop("quality", axis=1)
    y = df["quality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def create_model(input_size: int, hidden_layers: list) -> tf.keras.Model:
    """
    Opis:
        Tworzy model sieci neuronowej o zadanej architekturze.

    Parametry:
        input_size (int): Liczba cech wejściowych.
        hidden_layers (list): Lista neuronów w warstwach ukrytych.

    Zwraca:
        tf.keras.Model: Skonstruowany model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))

    for neurons in hidden_layers:
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))

    model.add(tf.keras.layers.Dense(11, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model: tf.keras.Model, X_train, y_train, X_test, y_test):
    """
    Opis:
        Trenuje model sieci neuronowej.

    Parametry:
        model (tf.keras.Model): Model sieci neuronowej.
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
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )


def create_confusion_matrix(model: tf.keras.Model, X_test, y_test):
    """
    Opis:
        Oblicza macierz pomyłek (confusion matrix).

    Parametry:
        model (tf.keras.Model): Wytrenowany model.
        X_test: Dane testowe.
        y_test: Rzeczywiste etykiety.

    Zwraca:
        ndarray: Macierz pomyłek.
    """
    predictions = np.argmax(model.predict(X_test), axis=1)
    return confusion_matrix(y_test, predictions)

def evaluate_model(model: tf.keras.Model, X_test, y_test) -> float:
    """
    Opis:
        Oblicza skuteczność (accuracy) wytrenowanego modelu
        na zbiorze testowym.

    Parametry:
        model (tf.keras.Model): Wytrenowany model sieci neuronowej.
        X_test: Dane testowe.
        y_test: Etykiety testowe.

    Zwraca:
        float: Skuteczność modelu (accuracy).
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


def main():
    """
    Opis:
        Funkcja główna programu realizująca pełny proces:
        pobranie danych, trenowanie modeli i ewaluację.
    """
    download_dataset(DATASET_URL, DATASET_PATH)

    data = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = process_data(data)

    small_model = create_model(X_train.shape[1], [16])
    large_model = create_model(X_train.shape[1], [64, 32])

    print("Trenowanie małej sieci neuronowej")
    train_model(small_model, X_train, y_train, X_test, y_test)

    print("Trenowanie większej sieci neuronowej")
    train_model(large_model, X_train, y_train, X_test, y_test)

    accuracy_small = evaluate_model(small_model, X_test, y_test)
    accuracy_large = evaluate_model(large_model, X_test, y_test)

    print("\nSkuteczność modeli:")
    print(f"Mała sieć neuronowa:  {accuracy_small:.4f}")
    print(f"Duża sieć neuronowa:  {accuracy_large:.4f}")

    cm = create_confusion_matrix(large_model, X_test, y_test)
    print("\nConfusion Matrix (duża sieć):")
    print(cm)


if __name__ == "__main__":
    main()
