# Sieci Neuronowe dla Klasyfikacji

## Wyniki eksperymentów

| Zbiór danych  | Model                | Skuteczność (accuracy) |
|---------------|--------------------|------------------------|
| Wine Quality  | Mała sieć          | 0.5344                 |
| Wine Quality  | Duża sieć          | 0.6062                 |
| CIFAR-10      | CNN                | 0.7193                 |
| Fashion-MNIST | CNN                | 0.9151                 |
| Mushroom      | MLP (64-32)        | 1.0000                 |

## Wnioski

1. **Wpływ wielkości sieci**  
   Większa sieć neuronowa (64-32) uzyskała wyższą skuteczność niż mała sieć (16 neuronów) przy klasyfikacji jakości wina. Pokazuje to, że większa liczba neuronów pozwala sieci lepiej dopasować się do danych. Jednak skuteczność sieci neuronowych była niższa niż klasycznych metod (drzewa decyzyjne, SVM), co może wynikać z niewielkiej liczby próbek i niezbalansowanych klas.

2. **Porównanie z poprzednimi metodami**  
   Drzewa decyzyjne i SVM osiągały skuteczność powyżej 0.85-0.90 dla Wine Quality, podczas gdy sieci neuronowe uzyskały tylko 0.6062. To pokazuje, że dla małych, tablicowych zbiorów danych klasyczne metody wciąż mogą być skuteczniejsze.

3. **Rozpoznawanie obrazów**  
   - CNN dobrze radzą sobie z klasyfikacją obrazów: CIFAR-10 (0.7193) i Fashion-MNIST (0.9151).  
   - Sieci neuronowe sprawdzają się lepiej dla większych i bardziej złożonych zbiorów obrazów.

4. **Perfekcyjna klasyfikacja grzybów (Mushroom Dataset)**  
   - Sieć neuronowa (MLP 64-32) osiągnęła **100% skuteczności** w klasyfikacji grzybów na jadalne i trujące.  
   - Zbiór Mushroom jest kompletny, dobrze opisany i w pełni kategoryczny, co umożliwia perfekcyjną klasyfikację.

5. **Ogólne obserwacje**  
   - Sieci neuronowe wymagają dużej ilości danych, aby osiągnąć wysoką skuteczność, szczególnie w przypadku obrazów.  
   - Dla małych, tablicowych danych klasyczne metody mogą nadal być konkurencyjne.  
   - Dla dobrze rozdzielonych, kategorycznych zbiorów danych, takich jak Mushroom, sieci MLP mogą osiągnąć perfekcyjną skuteczność.
