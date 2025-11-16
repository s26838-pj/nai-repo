from recommender import recommend_movies

if __name__ == "__main__":
    user_id = 1

    top, bottom = recommend_movies(user_id)

    print(f"Filmy dla użytkownika o id {user_id}")

    print("POLECAMY - 5 rekomendacji:\n")
    for title, score, plot in top:
        print(f"{title} — przewidywana ocena: {score}\nOpis: {plot}\n")

    print("NIE OGLĄDAĆ!!! 5 anty-rekomendacji:\n")
    for title, score, plot in bottom:
        print(f"{title} — przewidywana ocena: {score}\nOpis: {plot}\n")
