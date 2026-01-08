   
"""
    Temat: Prototyp maszyny do gry w "Baba Jaga patrzy".
    Autorzy: Robert Elwart, Szymon Stefański

    Wymagania i instalacja:
    Do działania programu wymagane jest zainstalowanie biblioteki oraz 'opencv-python'. Wpisz w terminalu:
    Instrukcja użycia:
    1. Zainstaluj bibliotekę: pip install opencv-python numpy.
    2. Uruchom skrypt (python main.py).    
    3. Stań przed kamerą. Poruszaj się, aby wywołać alarm i celownik na twarzy ("RUCH!"). 
       Zastygnij w bezruchu, aby zobaczyć zielony komunikat "STOP".
    4. Aby zamknąć program, kliknij na okno podglądu i naciśnij klawisz 'q'.

    Działanie:
    Uruchamia podgląd z kamery z funkcją wykrywania ruchu i twarzy (typu "Baba Jaga patrzy").

    1. Analizuje różnice między klatkami wideo, aby wykryć ruch.
    2. Jeśli ruch przekroczy próg (MOTION_THRESHOLD):
       - Uruchamia detekcję twarzy (Haar Cascade).
       - Rysuje celownik na wykrytej twarzy.
       - Wyświetla komunikat "RUCH!".
    3. Jeśli brak ruchu:
       - Wyświetla zielony komunikat "STOP".
    
    Sterowanie:
    - Klawisz 'q' lub 'ESC': Zamyka okno i kończy program.
    """

import cv2
import numpy as np
import sys

def main():
    """
    Opis:
        Uruchamia aplikację do detekcji ruchu i wykrywania twarzy
        w obrazie z kamery internetowej w czasie rzeczywistym.
    """
    
    # Ścieżka do klasyfikatora twarzy
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Sprawdzenie, czy plik XML został poprawnie załadowany
    if face_cascade.empty():
        print("Błąd: Nie znaleziono pliku haarcascade xml.")
        sys.exit()

    # Inicjalizacja kamery (indeks 0 to zazwyczaj domyślna kamera w laptopie)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        sys.exit()

    # Zmienne pomocnicze
    prev_gray = None
    MOTION_THRESHOLD = 50000  # Czułość wykrywania ruchu (im mniej, tym czulszy)

    print("--- Program uruchomiony ---")
    print("Naciśnij klawisz 'q' lub 'ESC', aby zakończyć.")

    try:
        while True:
            # Pobranie klatki
            ret, frame = cap.read()
            if not ret:
                print("Błąd: Nie można pobrać klatki wideo.")
                break

            # Konwersja na odcienie szarości i rozmycie (redukcja szumów)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            motion_detected = False

            # --- DETEKCJA RUCHU ---
            if prev_gray is not None:
                
                diff = cv2.absdiff(prev_gray, gray)
               
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
               
                motion_level = np.sum(thresh)

                if motion_level > MOTION_THRESHOLD:
                    motion_detected = True

            prev_gray = gray

            # --- LOGIKA APLIKACJI ---
            if motion_detected:
                
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                for (x, y, w, h) in faces:
                    cx = x + w // 2
                    cy = y + h // 2

                    # Rysowanie celownika
                    cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 2)
                    cv2.line(frame, (cx - 50, cy), (cx + 50, cy), (0, 0, 255), 2)
                    cv2.line(frame, (cx, cy - 50), (cx, cy + 50), (0, 0, 255), 2)

                    cv2.putText(frame, "RUCH!",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 3)
                    
                    
                    break 

            else:
                # Stan spoczynku
                cv2.putText(frame, "STOP",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)

            # Wyświetlenie okna
            cv2.imshow("Baba Jaga patrzy", frame)

            # --- OBSŁUGA WYJŚCIA ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:

        print("Zamykanie programu...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

