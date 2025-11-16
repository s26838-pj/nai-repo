"""
Projekt: Silnik rekomendacji filmów
Autorzy: Szymon Stefański, Robert Elwart

Opis:
    Program implementuje prosty silnik rekomendacji filmów na podstawie podobieństwa 
    użytkowników z wykorzystaniem miary podobieństwa kosinusowego.

    Funkcjonalności programu:
    - wczytuje oceny użytkowników z pliku ratings.csv,
    - wczytuje listę filmów z pliku movies.csv,
    - buduje macierz użytkownik–film,
    - oblicza podobieństwo użytkowników,
    - przewiduje oceny filmów nieoglądanych przez użytkownika,
    - zwraca 5 rekomendacji i 5 anty-rekomendacji,
    - pobiera dodatkowe informacje o filmie (opis) z zewnętrznego API TMDB.

Instrukcja przygotowania środowiska:
    1. Zainstaluj Python 3.x i dodaj go do zmiennej PATH.
    2. W IDE (np. PyCharm, VS Code) ustaw interpreter Pythona.
    3. W terminalu zainstaluj wymagane biblioteki: pip install pandas numpy requests scikit-learn python-dotenv
    4. Utwórz darmowe konto na stronie The Movie Database (TMDB) https://www.themoviedb.org/ i wygeneruj klucz API.
    5. W folderze projektu utwórz plik o nazwie: ".env" .
    6. W pliku ".env" dodaj zmienną środowiskową: TMDB_TOKEN
    7. Przypisz do zmiennej TMDB_TOKEN swój token ze strony.
"""


import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


load_dotenv()
TMDB_TOKEN = os.getenv("TMDB_TOKEN")


def get_movie_plot(title):
    """
    Pobiera opis filmu z TMDb API z obsługą polskich tytułów i dopasowaniem.

    Args:
        title (str): Tytuł filmu, który ma zostać wyszukany w TMDb.

    Returns:
        str: Opis filmu (po polsku lub angielsku). 
             Jeśli opis jest niedostępny, zwracany jest komunikat 
             o jego braku.
    """
    url = "https://api.themoviedb.org/3/search/movie"

    headers = {
        "Authorization": f"Bearer {TMDB_TOKEN}",
        "accept": "application/json"
    }

    params = {
        "query": title,
        "language": "pl-PL",
        "include_adult": "false"
    }

    response = requests.get(url, headers=headers, params=params).json()
    results = response.get("results", [])

    if not results:
        params = {
            "query": title,
            "include_adult": "false"
        }
        response = requests.get(url, headers=headers, params=params).json()
        results = response.get("results", [])

    if not results:
        return "Brak opisu w bazie TMDb."

    film = results[0]

    overview = film.get("overview", "").strip()

    if not overview:
        movie_id = film["id"]
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details = requests.get(details_url, headers=headers, params={"language": "en-US"}).json()
        overview = details.get("overview", "").strip()

    if not overview:
        return "Brak opisu w bazie TMDb."

    return overview
    

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

rating_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

rating_filled = rating_matrix.fillna(0)

similarity = cosine_similarity(rating_filled)
similarity_df = pd.DataFrame(
    similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)


def recommend_movies(target_user, n=5):
    """
    Generuje rekomendacje filmów na podstawie podobieństwa użytkowników.

    Parametry:
        target_user (int): ID użytkownika dla którego generujemy rekomendacje.
        n (int): liczba rekomendowanych i anty-rekomendowanych filmów.

    Zwraca:
        tuple(list, list):
            top_n — lista najlepszych rekomendacji z tytułami i opisami
            bottom_n — lista najsłabszych rekomendacji z tytułami i opisami
    """
    user_sim = similarity_df[target_user]
    user_ratings = rating_matrix.loc[target_user]

    unseen_movies = user_ratings[user_ratings.isna()].index

    scores = {}

    for movie in unseen_movies:
        total = 0
        sim_sum = 0

        for other_user in rating_matrix.index:
            rating = rating_matrix.loc[other_user, movie]

            if not np.isnan(rating):
                total += user_sim[other_user] * rating
                sim_sum += abs(user_sim[other_user])

        if sim_sum > 0:
            scores[movie] = total / sim_sum

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top5 = sorted_scores[:n]
    bottom5 = sorted_scores[-n:]

    top5_named = []
    bottom5_named = []

    for mid, score in top5:
        title = movies[movies.movieId == mid].title.values[0]
        plot = get_movie_plot(title)
        top5_named.append((title, round(score, 2), plot))

    for mid, score in bottom5:
        title = movies[movies.movieId == mid].title.values[0]
        plot = get_movie_plot(title)
        bottom5_named.append((title, round(score, 2), plot))

    return top5_named, bottom5_named
