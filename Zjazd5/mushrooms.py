"""
Projekt: Klasyfikacja grzybów (Mushroom Dataset)
Autorzy: Szymon Stefański, Robert Elwart

Opis:
    Program realizuje zadanie klasyfikacji grzybów
    na jadalne oraz trujące z wykorzystaniem
    sztucznej sieci neuronowej.

    Wykorzystano zbiór danych Mushroom Dataset,
    który zawiera cechy morfologiczne grzybów.

    Dane pochodzą ze strony:
    https://archive.ics.uci.edu/dataset/73/mushroom

Cele projektu:
    - nauczenie sieci neuronowej rozróżniania
      grzybów jadalnych i trujących,
    - zapoznanie się z klasyfikacją danych tabelarycznych,
    - realizacja zadania nr 4 (własny przypadek użycia).

Instrukcja uruchomienia:
    1. Zainstaluj Python 3.9 lub nowszy.
    2. W IDE (np. PyCharm, VS Code) ustaw interpreter Pythona.
    3. Zainstaluj biblioteki: pip install tensorflow pandas scikit-learn
    4. Uruchom program poleceniem: python mushrooms.py
"""

import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_mushroom_data():
    """
    Opis:
        Wczytuje zbiór danych Mushroom z repozytorium UCI.

    Zwraca:
        DataFrame: Dane grzybów.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises",
        "odor", "gill-attachment", "gill-spacing", "gill-size",
        "gill-color", "stalk-shape", "stalk-root",
        "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring",
        "veil-type", "veil-color", "ring-number", "ring-type",
        "spore-print-color", "population", "habitat"
    ]

    return pd.read_csv(url, header=None, names=columns)


def preprocess_data(df):
    """
    Opis:
        Koduje dane kategoryczne na wartości liczbowe
        oraz dzieli zbiór na dane treningowe i testowe.

    Parametry:
        df: DataFrame z danymi Mushroom.

    Zwraca:
        tuple: X_train, X_test, y_train, y_test
    """
    encoder = LabelEncoder()

    for column in df.columns:
        df[column] = encoder.fit_transform(df[column])

    X = df.drop("class", axis=1)
    y = df["class"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(input_size):
    """
    Opis:
        Tworzy prostą sieć neuronową typu MLP
        do klasyfikacji grzybów.

    Parametry:
        input_size: Liczba cech wejściowych.

    Zwraca:
        tf.keras.Model: Skonstruowany model.
    """
    model = models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(input_size,)))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Opis:
        Trenuje sieć neuronową na zbiorze Mushroom.

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
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )


def evaluate_model(model, X_test, y_test):
    """
    Opis:
        Oblicza skuteczność modelu na zbiorze testowym.

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
    df = load_mushroom_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = create_model(X_train.shape[1])

    print("Trenowanie sieci neuronowej dla Mushroom Dataset")
    train_model(model, X_train, y_train, X_test, y_test)

    accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nSkuteczność sieci neuronowej: {accuracy:.4f}")


if __name__ == "__main__":
    main()
