"""
Algorytm dynamicznego sterowania ruchem drogowym.

Program implementuje system Mamdanego do optymalizacji czasu zielonego światła na skrzyżowaniu 
dla kierunku Północ-Południe (NS) na podstawie 3 wejść: natężenia ruchu NS/EW, EW(wschód-zachód) (0-100 pojazdów/min) 
i średniego czasu oczekiwania (0-60 s, np. dla kolejki NS). Wyjście: czas zielonego dla NS (10-60 s). 

Autorzy: Szymon Stefański, Robert Elwart

Przygotowanie środowiska:
1. Zainstaluj Python 3.x i dodaj do PATH.
2. W IDE(np.PyCharm) ustaw interpreter Pythona.
3. Uruchom w terminalu: pip install numpy matplotlib.
4. Uruchom program - poprzez konsole wpisując: `python traffic_fuzzy.py` lub poprzez ręczne kliknięcie przycisku "Run".
5. Wyświetli wyniki symulacji i zapisze wykres do 'graph.png'.
"""
import numpy as np
import matplotlib.pyplot as plt

def triangular_membership(x, a, b, c):
    """

    Trójkątna funkcja przynależności - wektorowa.

    Parameters
    ----------

        x (float or np.ndarray): Wartość wejściowa (skalar lub tablica).
        a (float): Lewy punkt trójkąta (początek rosnącej krawędzi).
        b (float): Środkowy punkt (szczyt, przynależność = 1).
        c (float): Prawy punkt (koniec malejącej krawędzi).

    Returns
    -------
    np.ndarray
        Stopień przynależności mu(x) w zakresie [0, 1]

    """
    mu = np.zeros_like(x)
    mu = np.where((x > a) & (x <= b), (x - a) / (b - a), mu)
    mu = np.where((x > b) & (x < c), (c - x) / (c - b), mu)
    mu = np.where(x == b, 1.0, mu) 
    return mu

def get_membership_functions():
    """
    Generuje i zwraca słownik z funkcjami przynależności dla wejść i wyjścia.

    Zakresy: traffic (0-100), waiting (0-60), output (10-60).
    """
    # Traffic (0-100)
    def traffic_low(x): return triangular_membership(x, 0, 20, 40)
    def traffic_med(x): return triangular_membership(x, 20, 40, 60)
    def traffic_high(x): return triangular_membership(x, 50, 80, 100)
    
    # Waiting (0-60s)
    def wait_low(x): return triangular_membership(x, 0, 10, 20)
    def wait_med(x): return triangular_membership(x, 10, 30, 50)
    def wait_high(x): return triangular_membership(x, 40, 50, 60)
    
    # Output (10-60s)
    def out_low(x): return triangular_membership(x, 10, 20, 30)
    def out_med(x): return triangular_membership(x, 20, 30, 40)
    def out_high(x): return triangular_membership(x, 35, 50, 60)
    
    return {
        'traffic': {'Low': traffic_low, 'Medium': traffic_med, 'High': traffic_high},
        'waiting': {'Low': wait_low, 'Medium': wait_med, 'High': wait_high},
        'output': {'Low': out_low, 'Medium': out_med, 'High': out_high}
    }

def apply_rules(inputs, membership_funcs):
    """
    Zastosowanie logiki rozmytej Mamdaniego na podstawie 3 wejść.
    Oblicza stopnie przynależności wejść, stosuje 9 bazowych reguł z natężeń
    i modyfikatory z waiting, agregując do zbiorów wyjściowych (min-max).

    Args:
        inputs (list): Lista 3 wartości: [traffic_ns, traffic_ew, waiting_time].
        membership_funcs (dict): Słownik funkcji przynależności z get_membership_functions().
    
    Returns:
        dict: Stopnie przynależności wyjścia: {'Low': float, 'Medium': float, 'High': float}.
    """
    ns, ew, wait = inputs
    
    ns_l = membership_funcs['traffic']['Low'](ns)
    ns_m = membership_funcs['traffic']['Medium'](ns)
    ns_h = membership_funcs['traffic']['High'](ns) 

    ew_l = membership_funcs['traffic']['Low'](ew)
    ew_m = membership_funcs['traffic']['Medium'](ew)
    ew_h = membership_funcs['traffic']['High'](ew)
    
    w_l = membership_funcs['waiting']['Low'](wait)
    w_m = membership_funcs['waiting']['Medium'](wait)
    w_h = membership_funcs['waiting']['High'](wait)
    
    
    r1 = min(ns_l, ew_l)  # Low
    r2 = min(ns_l, ew_m)  # Low
    r3 = min(ns_l, ew_h)  # Med
    r4 = min(ns_m, ew_l)  # Med
    r5 = min(ns_m, ew_m)  # Med
    r6 = min(ns_m, ew_h)  # Low
    r7 = min(ns_h, ew_l)  # High
    r8 = min(ns_h, ew_m)  # Med
    r9 = min(ns_h, ew_h)  # Low

    base_low = max(r1, r2, r6, r9)
    base_med = max(r3, r4, r5, r8)
    base_high = r7
    
    """
    Agregacja wyjść (max dla OR)
    """
    boost_high = min(ns_h, w_h)  # Wysokie NS + wysokie wait -> High
    boost_med = min(ns_m, w_h)   # Średnie NS + wysokie wait -> Med
    reduce_low = min(w_h, ew_h)  # Wysokie wait + wysokie EW -> redukcja Low (priorytet NS)

    output_low = max(base_low, reduce_low) * max(0, 1 - w_h)  # Redukuj Low przy wysokim wait
    output_med = max(base_med, boost_med)
    output_high = max(base_high, boost_high)
    
    return {'Low': float(output_low), 'Medium': float(output_med), 'High': float(output_high)}

def defuzzify(output_membership, membership_funcs, num_points=100):
    """
    Defuzyfikacja metodą centroidu (środka ciężkości).

    Buduje powierzchnię agregowaną i oblicza centroid.

    Args:
        output_membership (dict): Stopnie przynależności wyjścia z apply_rules().
        membership_funcs (dict): Słownik funkcji przynależności.
        num_points (int): Liczba punktów do dyskretyzacji (domyślnie 100).
    
    Returns:
        float: Defuzyfikowana wartość czasu zielonego (w sekundach).
    
    """
    x = np.linspace(10, 60, num_points)
    y = np.zeros_like(x)
    
    y = np.maximum(y, np.minimum(output_membership['Low'], membership_funcs['output']['Low'](x)))
    y = np.maximum(y, np.minimum(output_membership['Medium'], membership_funcs['output']['Medium'](x)))
    y = np.maximum(y, np.minimum(output_membership['High'], membership_funcs['output']['High'](x)))
    
    if np.sum(y) == 0:
        return 10.0
    
    centroid = np.trapezoid(x * y, x) / np.trapezoid(y, x)
    return centroid

class TrafficFuzzyController:
    """
    Klasa kontrolera rozmytego do sterowania ruchem drogowym.

    Używa systemu Mamdani z 3 wejściami do obliczania czasu zielonego światła.
    """

    def __init__(self):
        """
        Inicjalizuje kontroler i ładuje funkcje przynależności.
        """
        self.membership_funcs = get_membership_functions()
    
    def compute_green_time(self, traffic_ns, traffic_ew, waiting_time):
        """
        Oblicza optymalny czas zielonego dla kierunku NS.

        Args:
            traffic_ns (float): Natężenie ruchu na NS (0-100 pojazdów/min).
            traffic_ew (float): Natężenie ruchu na EW (0-100 pojazdów/min).
            waiting_time (float): Średni czas oczekiwania (0-60 s).
        
        Returns:
            float: Czas zielonego dla NS (10-60 s).
        """
        inputs = [traffic_ns, traffic_ew, waiting_time]
        output_mf = apply_rules(inputs, self.membership_funcs)
        green_time = defuzzify(output_mf, self.membership_funcs)
        return green_time

"""
Symulacja działania systemu sterowania ruchem
"""
if __name__ == "__main__":
    controller = TrafficFuzzyController()
    cycle_time = 60

    scenarios = [
        (20, 20, 5),    # Niskie wszystko
        (30, 70, 40),   # Średni NS, wysoki EW, wysoki wait
        (80, 10, 20),   # Wysoki NS, niski EW, średni wait
        (50, 50, 50)    # Średnie natężenia, wysoki wait
    ]

    print("Symulacja sterowania ruchem (3 wejścia):")
    print("Natężenie NS | Natężenie EW | Czas oczek. | Zielone NS (s) | Zielone EW (s)")
    print("-" * 70)

    for ns, ew, wait in scenarios:
        green_ns = controller.compute_green_time(ns, ew, wait)
        green_ew = cycle_time - green_ns  
        print(f"{ns:12} | {ew:13} | {wait:11} | {green_ns:13.2f} | {green_ew:13.2f}")

    # Wizualizacja - zapisanie do pliku
    def plot_membership(mfs, axs, title, ranges):
        # Traffic (wspólny dla NS/EW)
        x_traffic = np.linspace(*ranges[0], 100)
        for label, func in mfs['traffic'].items():
            y = func(x_traffic)
            axs[0].plot(x_traffic, y, label=label)
        axs[0].set_title('Wejście: Natężenie (0-100)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Waiting
        x_wait = np.linspace(*ranges[1], 100)
        for label, func in mfs['waiting'].items():
            y = func(x_wait)
            axs[1].plot(x_wait, y, label=label)
        axs[1].set_title('Wejście: Czas oczek. (0-60s)')
        axs[1].legend()
        axs[1].grid(True)
        
        # Output
        x_out = np.linspace(10, 60, 100)
        for label, func in mfs['output'].items():
            y = func(x_out)
            axs[2].plot(x_out, y, label=label)
        axs[2].set_title('Wyjście: Czas zielonego (10-60s)')
        axs[2].legend()
        axs[2].grid(True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_membership(controller.membership_funcs, axs, '', [(0,100), (0,60)])
    plt.tight_layout()
    plt.savefig('graph.png')
    print("\nWykres zapisany")



    