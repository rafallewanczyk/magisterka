# Przegląd literatury

## Rozpoznawanie dyscypliny sportu na podstawie materiału wideo

### Wprowadzenie (str. 2)

Rozpoznawanie akcji można podzielić na kilka rodzai:

- klasyfikacja akcji (analiza fragmentu wideo zawierającego pojedynczą akcję)
    - klasyfikacja akcji wykonywanej przez 1 osobę
    - klasyfikacja akcji grupowej (można wydzielić pojedyncze osoby, wchodzące ze sobą w interakcje) np. bójka, sporty
      drużynowe
    - klasyfikacja akcji tłumu (nie da się wydzielić pojedynczych osób) np. demonstracja
- wykrywanie akcji (znalezienie segmentu wideo zawierającego akcje, następnie sklasyfikowanie go)

Większość wcześniej przeprowadzonych badań skupiało się głównie na pierwszych dwóch rodzajach klasyfikacji akcji.
Rozpoznawanie interakcji w ostatnich latach przyciąga coraz większą uwagę, grupowe rozpoznawanie akcji jest we wczesnym
stadium badań

### Wyzwania (str. 1)

- W grupowej klasyfikacji aktywności, akcje pojedynczych osób są niejednoznaczne np. stanie w kolejce, jest czymś innym
  niż staniem w grupie i rozmawianiem
- Główną informację może nieść niewielka ilość osób w grupie będąca otoczona innymi osobami niosącymi mniejszą ilość
  informacji lub zakłócającymi całkowity obraz. Należy rozważyć cały kontekst sytuacji. np. w meczu siatkówki atakująca
  oraz blok ruszają się bardzo dynamicznie, a reszta zawodników stoi w gotowości
- Zamodelowanie akcji wykonywanej przez człowieka
- Reprezentacja cech modelu (cechy reprezentujące wygląd i poza człowieka nie wystarczają, zamodelować należy również
  ich zmianę w czasie. Dodany zostaje kolejny wymiar reprezentujący czas)

### Cechy zawarte w danych

- dane RGB
    - cechy:
        - czasoprzestrzenne objętościowe cechy (spatiotemporal volume-based features
        - czasoprzestrzenne cechy punktów zainteresowań
        - cechy śledzenia stawów
    - ograniczenia:
        - ruch kamery
        - ograniczenia detekcji człowieka
- dane o głębi RGBD (wymaga specjalnych kamer)
    - cechy takie same jak RGB
    - zastosowanie głębi eliminuje problemy związane ze zmianami w otoczeniu
- deep learning
    - model sam wyucza się najlepszych cech (cechy te w dużej mierze są dopasowane do realizowanego zadania, są trudne
      lub niemożliwe do zinterpretowania przez człowieka)

### Metody rozpoznawania akcji przy ręcznym modelowaniu cech

#### Metoda z góry do dołu (Top-Down Approach)

Skupiona na analizie globalnych wzorców ruchu całej grupy oraz śledzenie trajektorii, oraz interakcji w całej grupie.
Akcje pojedynczego uczestnika w grupie są mniej ważne.

##### Metody oparte na trajektoriach

Analiza grupowych aktywności pod względem interakcji pomiędzy pojedynczymi trajektoriami. Vaswani[37] modeluje
zachowania grupowe poprzez reprezentowanie ruszających się obiektów jako punkty w dwu-wymiarowej płaszczyznie. W ten
sposób grupowe zachowanie przedstawione jest jako zmieniający się w czasie wielokąt. Akcja zostaje przedstawiona jako
średnia ze wszystkich kształtów w przestrzeni tangensowej.

Cheng et al. [12] zaproponował proces Gaussowski klasyfikacji aktywności. Proces polega na prześledzeniu trajektorii
każdej z osób. Następnie zastosowany zostaje regresja procesu gaussowskiego oraz informacje dotyczące ruchu zostają
obliczone na podstawie trzech zaproponowanych współczynników - wzorca indywidualnego, dualnego oraz globalnego.
Następnie zastosowana zostaje metoda Bag-Of-Words w celu uporządkowania wyznaczonych cech. Predykcja odbywa się za
pomocą modelu SVM

##### Metody oparte o podgrupy

W celu uproszczenia sytuacji zaproponowane zostały metody polegające na podział grupy wykonującej aktywność na podgrupy
oraz szukania zależności pomiędzy podgrupami. Podział na podgrupy ułatwia zadanie klasyfikacji, jednak dokonanie
poprawnego podziału jest bardzo trudne.

Zhang et al. [13] zaproponował podział na podgrupy na podstawie wielogrupowej przyczynowości. Wyznaczony został podział
udziałowość indywidualną, w parze, między-grupową, ogólną. Każdy z podziałów został zakodowany przy pomocy kodowania
LCC, budując zbiór cech modelu SVM.

#### Metoda z dołu do góry (Down-Top Approach)

Skupiona na rozpoznawaniu pojedynczych osób następnie analizowanie ich struktury hierarchalnej na poziomie pojedynczej
osoby oraz poziomie grupy.

##### Metody oparte na deskryptorach

Metody te łączą ze sobą koteksty sceny poprzez deskryptory wyznaczające kluczowe elementy scen. Choi et al.[9]
przedstawia lokalnie czasowo-przestrzenny deskryptor (STL), który określa czasowo-przestrzenną dystrybucję pozycji,
pozy, oraz informacji o ruchu pojedynczych osób. Osoby oraz ich sylwetki są wykrywane poprzez zastosowanie histogramów
zorientowanych gradientów (HOG) oraz modelu SVM.

Lan et al. [31] wykorzystał informację mówiącą, że akcje osób wokół klasyfikowanej aktywności niesie dużo informacji o
akcjach pojedynczych osób, wyznaczył deskryptor opisujący akcję głównej osoby nazwanej zakotwiczoną, oraz wszystkich
osób wokół niej. Badania pokazały, że deskryptor ten bardzo dobrze sprawdza się przy nadzorowaniu aktywności grupy,
jednak jest on zbyt czuły na położenie kamery.

##### Metody oparte na kontekście niesionym przez interakcję

Metody te opierają się na obserwacji, mówiącej, że dużą ilość informacji o aktywności grupowej niosą interakcje pomiędzy
uczestnikami danej sytuacji (podawanie ręki, uściski). Len at el. [34] przedstawił hierarchiczny model interakcji oraz
adaptacyjny mechanizm służący do automatycznego wykrywania pasujących do modelu interakcji. Wykrywanie interakcji bazuje
na budowie grafu aktywności, dzielącego uczestników na podgrupy oraz na podstawie ich zachowań, znajdowaniu zachodzącej
akcji.

### Metody Deeplearningu (automatyczna nauka cech)

W ostatnich latach splotowe sieci neuronowe (CNN) osiągnęły bardzo dobre wyniki w zadaniach percepcji maszyn, takich jak
rozpoznawanie obrazu, rozpoznawanie elementów na obrazie lub klasyfikacja wideo. Po zastosowaniu sieci CNN w zadaniach
klasyfikacji aktywności grupowej osiągnęły one znacznie lepsze wyniki od metod używających ręcznie wyznaczone cechy
modelu.

#### Hierarchiczne modelowanie czasowe

Przy klasyfikacji grupowej aktywności należy śledzić zmiany stanów kilku osób. Dużym wyzwaniem jest konstrukcja modelu,
który będzie w stanie skupić się na zmianach w położeniu i czasie kilku śledzonych osób. Ibrahim et al. zaproponował
dwu-poziomowy model korzystający z rekurencyjnej sieci LSTM. Pierwszy poziom modelu składa się z sieci grupy sieci LSTM,
pojedyncza sieć LSTM odpowiada jednej osobie biorącej udział w aktywności, oraz karmiona jest danymi dotyczącymi zmian
jej stanów. Na drugim poziomie modelu wyjścia z pojedynczych sieci są agregowane oraz wykorzystane jako wejście do
kolejnej sieci LSTM, która wykonuje predykcje dotyczącą aktualnej aktywności.

#### Głębokie modelowanie związków

Znajdowanie związków pomiędzy akcjami uczestników aktywności jest kluczowe dla dobrej klasyfikacji. Znajdowanie tego
typu relacji jest skomplikowane ze względu na brak dostępnych danych, zbiory treningowe zawierają albo informacje
dotyczące aktywności pojedynczych osób w grupie lub całej grupy. Dużo badań poszukuje sposobu na odnalezienie metody
znajdywania relacji pomiędzy uczestnikami aktywności. Qi et al. [61] zaproponował stagNet - rekurencyjną sieć neuronową
wyposażoną w atencję. Rozwiązanie buduje graf semantyczny, używając etykiet słownych oraz danych wizualnych. Relacja
przestrzenna pomiędzy pojedynczymi uczestnikami akcji jest wnioskowana w grafie semantycznym na podstawie mechanizmu
przekazywania wiadomości. Ponad to model śledzi kluczową osobę w akcji.

##### Modele z uwagą

W większości aktywności najwięcej informacji niesie jedna lub kilka osób wykonywujących akcję, pozostałe osoby mogą
powodować zakłócenie modelu. Ze względu na bark informacji w zbiorach danych określających kluczową osobę, problem ten
jest dużym wyzwaniem. Ramanathan et al. [11] zastosował model z czasową uwagą na zbiorze danych skłądającym się z meczów
z koszykówki, znajdując w scenie kluczowych zawodników i poprawiając odczyt aktualnego zagrania.

Yan et al. [60] zauważył, że najczęściej kluczowe osoby to te, które poruszają się ruchem jednostajnym. W celu
znalezienia tego typu osób zastosowano przepływ optyczny. Na śledzonych osobach zastosowano sieci LSTM. Ważona suma
sieci LSTM użyta została jako wyjście z modelu. 





