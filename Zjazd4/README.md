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
- Pełna interaktywna aplikacja webowa (Streamlit) – dowolny `random_state`, wyniki i wykresy na żywo  
- Najlepsze wartości podświetlone na zielono  

**Źródła danych:**
- https://archive.ics.uci.edu/dataset/186/wine+quality
- https://archive.ics.uci.edu/dataset/34/diabetes

---

## Sposób 1 – Uruchomienie w terminalu (3 scenariusze)

```bash
# 1. Skopiuj i rozpakuj projekt, wejdź do folderu
cd Zjazd4

# 2. Utwórz i aktywuj wirtualne środowisko
python -m venv .venv
.\.venv\Scripts\activate     # Windows

source .venv/bin/activate  # Linux

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Uruchom analizę (wykona się automatycznie 3 scenariusze)
python main.py

```
---

## Sposób 2 – Aplikacja webowa (Streamlit)

```bash
# 1. Skopiuj i rozpakuj projekt, wejdź do folderu
cd Zjazd4

# 2. Utwórz i aktywuj wirtualne środowisko
python -m venv .venv
.\.venv\Scripts\activate     # Windows

source .venv/bin/activate  # Linux

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Uruchom analizę (wykona się automatycznie 3 scenariusze)
streamlit run app.py

# 5. Wybierz własny scenariusz
