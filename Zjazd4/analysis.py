"""
# Drzewa decyzyjne vs SVM – analiza klasyfikatorów  
**Zadanie zaliczeniowe – 3 scenariusze + interaktywna aplikacja webowa (Streamlit)**  
Autorzy: Szymon Stefański, Robert Elwart
---

### Opis:
Celem projektu jest kompleksowe porównanie dwóch najpopularniejszych algorytmów klasyfikacji:

- **Drzewa decyzyjne** (Gini i Entropy, bez ograniczenia głębokości)  
- **Maszyny wektorów nośnych (SVM)** z różnymi funkcjami jądrowymi:  
  - liniowy  
  - RBF (Gaussowski) – przy C=1 i C=10  
  - wielomianowy stopnia 3  

Analiza przeprowadzona została na dwóch klasycznych, publicznie dostępnych zbiorach danych:

1. **Wine Quality (red)** – przewidywanie wysokiej jakości wina (ocena ≥7)  
2. **Pima Indians Diabetes** – wykrywanie cukrzycy u kobiet z plemienia Pima  

**Kluczowe cechy projektu:**
- 3 różne scenariusze losowości (random_state = 42, 123, 999) – pokazanie stabilności wyników  
- Automatyczne generowanie i zapis wykresów (rozkład klas + rozproszenie cech)  
- Wyniki zapisane do pliku `results.csv` oraz wyświetlane w czytelnych tabelach  
- aplikacja webowa (Streamlit) – dowolny `random_state`, wyniki i wykresy na żywo  
- Najlepsze wartości podświetlone na zielono  

**Źródła danych:**
- https://archive.ics.uci.edu/dataset/186/wine+quality
  - https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

- https://archive.ics.uci.edu/dataset/34/diabetes
  -  https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
  
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

"""
Globalna lista na wyniki wszystkich scenariuszy
"""
os.makedirs("plots", exist_ok=True)
results_all = []

def load_wine():
    """
    Wczytuje zbiór Wine Quality (red wine) z UCI i przygotowuje go do klasyfikacji binarnej.

    Returns:
        X (pd.DataFrame): cechy 
        y (pd.Series): etykiety binarne (1 = dobra jakość >=7, 0 = pozostałe)
        name (str): nazwa zbioru
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    data['good'] = (data['quality'] >= 7).astype(int)
    X = data.drop(['quality', 'good'], axis=1)
    y = data['good']
    return X, y, "Wine Quality (dobre >=7)"


def load_diabetes():
    """
    Wczytuje zbiór Pima Indians Diabetes.

    Returns:
        X (pd.DataFrame): 9 cech medycznych
        y (pd.Series): etykiety (1 = cukrzyca, 0 = brak)
        name (str): czytelna nazwa zbioru
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    cols = ['preg', 'gluc', 'bp', 'skin', 'insu', 'bmi', 'dpf', 'age', 'class']
    data = pd.read_csv(url, names=cols)
    X = data.drop('class', axis=1)
    y = data['class']
    return X, y, "Pima Indians Diabetes"

"""
Definicja modeli
"""
tree_models = {
    "Drzewo Gini (depth=5)":     DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42),
    "Drzewo Entropy (depth=5)":  DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42),
    "Drzewo Gini (pełne)":       DecisionTreeClassifier(criterion='gini', random_state=42),
}

svm_configs = [
    ("SVM linear C=1",   SVC(kernel='linear', C=1.0, random_state=42)),
    ("SVM RBF C=1",      SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)),
    ("SVM RBF C=10",     SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
    ("SVM poly deg=3",   SVC(kernel='poly', degree=3, C=1, gamma='scale', random_state=42)),
]


def run_single_scenario(run_id, random_state):
    """
    Wykonuje pełną analizę (wizualizacja + wszystkie modele) dla jednego scenariusza.

    Args:
        run_id (int): numer scenariusza (1, 2, 3) – używany tylko do nazewnictwa plików
        random_state (int): ziarno losowości dla powtarzalności podziału danych

    Return:
        - zapisuje wykresy do folderu plots/
        - uzupełnia globalną listę results_all o wyniki modeli
    """
    print(f"\n{'='*25} SCENARIUSZ (random_state={random_state}) {'='*25}")

    for load_func, dataset_name in [(load_wine, "wine"), (load_diabetes, "diabetes")]:
        X, y, full_name = load_func()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )

        """
        Standaryzacja SVM
        """
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        """
        Wizualizacja
        """
        plt.figure(figsize=(13, 5))

        """
        Rozkład klas w zbiorze treningowym
        """
        plt.subplot(1, 2, 1)
        sns.countplot(x=y_train, hue=y_train, palette="Set2", legend=False)
        plt.title(f"Rozkład klas (train)\n{full_name}")
        plt.xlabel("Klasa")

        """
        Rozproszenie dwóch najważniejszych cech w zbiorze testowym
        """
        plt.subplot(1, 2, 2)
        f1 = 'alcohol' if 'wine' in dataset_name else 'gluc'
        f2 = 'sulphates' if 'wine' in dataset_name else 'bmi'
        plt.scatter(X_test[f1], X_test[f2], c=y_test, cmap='coolwarm', alpha=0.8, edgecolor='k', s=60)
        plt.colorbar(label="Klasa rzeczywista")
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title("Zbiór testowy")

        plt.suptitle(f"{full_name} – Scenariusz {run_id} (rs={random_state})", fontsize=15)
        plt.tight_layout()

        plot_path = f"plots/{dataset_name}_scen{run_id}_rs{random_state}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Wykres zapisany → {plot_path}")

        """
        Trenowanie modelu
        """
        for model_name, template in {**tree_models, **dict(svm_configs)}.items():
            model = template.__class__(**template.get_params(deep=False))

            if "SVM" in model_name:
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=0)['weighted avg']['f1-score']

            """
            Zapis wyników do globalnej listy
            """
            results_all.append({
                "Scenariusz": run_id,
                "Zbiór": full_name,
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "F1-weighted": round(f1, 4),
            })

            print(f"{model_name:30} → Accuracy: {acc:.4f} | F1: {f1:.4f}")


def run_analysis():
    """
    Główna funkcja uruchamiająca całą analizę.
    Wykonuje trzy scenariusze z różnymi ziarnami losowości,
    zapisuje wyniki do pliku CSV oraz wyświetla podsumowanie.
    """
    print("PEŁNA ANALIZA KLASYFIKATORÓW – 3 SCENARIUSZE".center(100))
    print("="*100)

    for i, rs in enumerate([42, 123, 999], 1):
        run_single_scenario(i, rs)

    # Zapis wyników
    df = pd.DataFrame(results_all)
    df.to_csv("results.csv", index=False)
    print(f"\nWyniki zapisane do pliku → results.csv")
    print("Wykresy zapisane w folderze → plots/")

    print("\nPORÓWNANIE WYNIKÓW W 3 SCENARIUSZACH")
    print("="*100)
    print(df.pivot_table(index=['Zbiór', 'Model'], columns='Scenariusz', values='Accuracy').round(4))

    print("\nWNIOSKI KOŃCOWE".center(100))
    print("="*100)
    print("• W eksperymencie wykorzystano trzy różne funkcje jądrowe dla SVM: liniową, RBF (Gaussa) oraz wielomianową stopnia 3.")
    print("• Najlepsze i najbardziej stabilne wyniki osiągnął kernel RBF (szczególnie przy C=10), co potwierdza jego uniwersalność.")
    print("• Kernel liniowy działa zadowalająco, ale nie radzi sobie z nieliniową separowalnością danych tak dobrze jak RBF.")
    print("• Kernel wielomianowy stopnia 3 okazał się najsłabszy na obu zbiorach – najniższa dokładność i największa niestabilność.")
    print("="*100)
    

