"""
Gra:
Pawn Duel

Zasady:
Uproszczona gra dwuosobowa na planszy 4x4.
Białe pionki (B) idą w górę, Czarne pionki (C) idą w dół.
Pionki mogą zostać zbite po skosie - pionek zbijający wtedy zajmuje miejsce zbitego pionka.
Wygrywa gracz, który dotrze do końca kolumny, zbije wszystkie piony przeciwnika
albo gdy przeciwnik nie może wykonać ruchu.

Autorzy: Szymon Stefański, Robert Elwart

Przygotowanie środowiska:
1. Zainstaluj Python 3.x i dodaj do PATH.
2. W IDE(np.PyCharm) ustaw interpreter Pythona.
3. Zainstaluj pakiet easyAI: w terminalu PyCharm wpisz `python -m pip install easyAI` lub ręcznie można importować.
4. Uruchom grę - poprzez konsole wpisując: `python PawnDuel.py` lub poprzez ręczne kliknięcie przycisku "Run".
"""

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class PawnDuel(TwoPlayerGame):
    """
    Klasa reprezentująca grę Pawn Duel w systemie easyAI.
    """

    def __init__(self, players=None):
        """
        Inicjalizuje grę z początkową planszą i ustawia gracza rozpoczynającego.

        Parameters:
            players (list): Lista graczy (np. [Human_Player(), AI_Player()])
        """
        self.players = players
        self.board = [
            ["C", "C", "C", "C"],
            [".", ".", ".", "."],
            [".", ".", ".", "."],
            ["B", "B", "B", "B"]
        ]
        self.current_player = 1  # Białe zaczynają

    def possible_moves(self):
        """
        Generuje wszystkie możliwe ruchy dla aktualnego gracza.

        Returns:
            list[str]: Lista możliwych ruchów w formacie "r1 c1 r2 c2"
        """
        p = "B" if self.current_player == 1 else "C"
        d = -1 if p == "B" else 1
        o = "C" if p == "B" else "B"
        moves = []

        for r in range(4):
            for c in range(4):
                if self.board[r][c] == p:
                    nr = r + d
                    if 0 <= nr < 4:
                        if self.board[nr][c] == ".":
                            moves.append(f"{r} {c} {nr} {c}")
                        if c > 0 and self.board[nr][c - 1] == o:
                            moves.append(f"{r} {c} {nr} {c - 1}")
                        if c < 3 and self.board[nr][c + 1] == o:
                            moves.append(f"{r} {c} {nr} {c + 1}")
        return moves

    def make_move(self, move):
        """
        Wykonuje ruch na planszy.

        Parameters:
            move (str): Ruch w formacie "r1 c1 r2 c2"
        """
        r1, c1, r2, c2 = map(int, move.split())
        self.board[r2][c2], self.board[r1][c1] = self.board[r1][c1], "."

    def win(self):
        """
        Sprawdza, czy aktualny gracz wygrał.

        Returns:
            bool: True jeśli aktualny gracz wygrał, False w przeciwnym razie.
        """
        f = sum(self.board, [])
        if "B" not in f:
            return self.current_player == 2
        if "C" not in f:
            return self.current_player == 1

        # Sprawdzenie dojścia do końca
        if "B" in self.board[0]:
            return self.current_player == 1
        if "C" in self.board[3]:
            return self.current_player == 2
        return False

    def is_over(self):
        """
        Sprawdza, czy gra się zakończyła (ktoś wygrał lub brak ruchów).

        Returns:
            bool: True jeśli gra się skończyła, False w przeciwnym razie.
        """
        return self.win() or not self.possible_moves()

    def scoring(self):
        """
        Funkcja oceny dla algorytmu AI.

        Returns:
            int: Ocena stanu gry (1 - wygrana, -1 - przegrana, 0 – neutralny stan).
        """
        return 100 if self.win() else 0

    def show(self):
        """
        Wyświetla aktualny stan planszy w konsoli.
        """
        print("\nAktualna plansza:")
        for row in self.board:
            print(" ".join(row))
        print()


if __name__ == "__main__":
    # Ustawienia AI
    ai_algo = Negamax(3)
    game = PawnDuel([Human_Player(), AI_Player(ai_algo)])
    game.play()

    if game.win():
        print(f"\nWygrał gracz {game.current_player} ({'Białe' if game.current_player == 1 else 'Czarne'})!")
