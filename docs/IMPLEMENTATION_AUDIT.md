# Audyt kodu i danych dla rozdziału 5

## Zakres migracji

Źródło: `mtl-emotion-intensity-sentiment`, commit `b5d8aaae5266229970b641f0b3b2e6e6da8607ea`, `EMOTIA-ML/multi_emotion_sentiment_intensity_classifier.py`. Klasy modeli, datasetu i FocalLoss skopiowano jako definicje AST do `models/inherited.py`. `docs/provenance.json` zawiera linie i SHA-256 każdej definicji; test sprawdza zgodność. `SoftSharingModel` występował dwukrotnie: migrowana jest ostatnia, faktycznie wiążąca definicja. Adapter w `models/__init__.py` ujednolica interfejs historycznych wyjść i umożliwia ablacje bez zmiany kodu odziedziczonych klas.

## Potwierdzone problemy i poprawki

| Problem historyczny | Nowy protokół |
|---|---|
| Intensywność oceniana/uczoma także dla emocji nieobecnych, gdzie brak był kodowany jako low | `-100` dla nieobecnych/brakujących; strata i metryki liczone tylko dla obecnych z poprawną etykietą. |
| Emotion F1 liczone po spłaszczeniu macierzy, co myliło micro-F1 z accuracy po etykietach | Multilabel micro/macro F1, subset accuracy i label accuracy raportowane oddzielnie. |
| Wagi emocji ustalane z całego zbioru | Wagi wyłącznie z train danego podziału. |
| Niespójne progi emocji, fallback tekstu i liczenie kroków gradient accumulation | Próg dobierany na dev i zamrażany przed testem; brakujący tekst augmentacji jest błędem; scheduler liczy rzeczywiste kroki optymalizatora. |
| Różne formaty wyjścia starych modeli STL/MTL | Adapter zwraca mapę zadanie → logits i maskuje nieaktywne głowy. |
| Jedna realizacja losowości i brak diagnostyk transferu | Pięć seedów w macierzy, zapis strat, gradientów i routingu na stałej próbce train. |

Powyższe zmiany modyfikują estymand względem starych tabel. Porównania w nowym protokole nie zastępują automatycznie historycznych liczb.

## MEISD i istniejące augmentacje

Plik one-hot: SHA-256 `458add484602056689a64079385b47a01f3bdba1a9531a33ba23880b1934c4ad`, 4219 wierszy (1611 oryginalnych, 2608 `llm`). Każdy z 2608 wierszy `llm` ma puste `Utterances`; brak w nim kolumny z tekstem augmentacji. Historyczny loader używał wówczas `original`, więc te wiersze powielały teksty rodziców. Kopia `(1)` ma identyczny hash.

Dodany `MEISD_balanced_expanded.csv`: SHA-256 `20974b3232d2dd1e903d5b333e6267a826eb4845cb922d3a744e84f045e49019`. Ma tę samą kolejność 4219 wierszy; wszystkie 2608 tekstów `augmented` są obecne. `Utterances`, `original` i `mode` są identyczne w każdym odpowiadającym wierszu. `sentiment` różni się w 83 wierszach wyłącznie poprawką `positve` → `positive`. Różne formaty/zaokrąglenia `quality` nie są traktowane jako nowe etykiety. Wszystkie rodzice augmentacji znaleziono wśród oryginałów. 142 teksty augmentacji są identyczne z własnym `original`; etap przygotowania usuwa ich identyczne kopie.

W 396 wierszach ta sama emocja zajmuje kilka slotów. W 216 z nich jej intensywności są sprzeczne; konwersja one-hot zachowała jedną, arbitralnie wybraną wartość. Adapter zachowuje obecność emocji i maskuje tylko jej niejednoznaczną intensywność (`-100`).

Surowe `MEISD_text.csv`: SHA-256 `30e5e212a2bf390af6d5fa01e8ac211072073a93b32c209917e276ca3957c749`. Odzyskano 1611/1611 oryginalnych połów dialogów i przypisano je do 980 dialogów. Podział po połączonych grupach dialogów i identycznych tekstach jest ustalony seedem 2026, bez dobierania pod wyniki. Sześć unikalnych tekstów ma sprzeczne etykiety, więc wykluczono 14 rekordów. Z dev/test usunięto 748 augmentacji, a po podziale odrzucono 102 identyczne kopie. Przygotowany zbiór `meisd-v2` ma 2877 train (1744 augmentacje), 231 dev i 247 test. Manifest danych zawiera komplet hashy i liczników.

Ograniczenia: historyczne filtrowanie/uzupełnianie etykiet już zaszło przed dostarczonym CSV. Intensywności połów dialogów były uśredniane według pozycji anotacji, która nie musi odpowiadać tej samej emocji. Teksty `llm` dziedziczą etykiety rodziców; nie przeprowadzono niezależnej oceny zgodności znaczenia i etykiet. Wyniki na tym zbiorze trzeba interpretować z tą niepewnością.

## BRIGHTER English

Oficjalny [zbiór Track B](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities), rewizja `08a5d61d3fc33036f97c8c76a61ff8d6f02f5157`, oryginalne liczby 2763 train, 115 dev, 2765 test. Anger, fear, joy, sadness i surprise mają etykiety 0–3. Obecność emocji jest wyprowadzona z intensywności >0, a nie osobno anotowana. Sentyment nie jest dostępny. Historyczne architektury służą tu porównaniu STL i MTL dwóch zadań.

Audyt wykazał powtarzające się identyczne teksty z różnymi etykietami w train/test, w tym `#NAME?`, oraz dwa teksty wspólne dla train i test. Wszystkie rekordy z konfliktem dokładnego tekstu wykluczono: 8 train i 6 test. Dwie zgodne kopie w train usunięto. Pozostało 2753/115/2759; zachowano członkostwo w oficjalnych splitach dla wszystkich pozostałych rekordów. Bez tego filtrowania test zawierałby teksty widziane podczas uczenia. Zmiana liczności względem oficjalnego benchmarku musi być jawna przy porównywaniu z jego tabelami.

Między przygotowanymi zbiorami MEISD i BRIGHTER nie znaleziono identycznych tekstów po normalizacji białych znaków i wielkości liter. Nie wyklucza to bliskich parafraz. Zewnętrzne badanie polega na osobnym treningu na BRIGHTER; nie jest testem zero-shot modelu wytrenowanego na MEISD. Oficjalna metryka Track B to korelacja Pearsona; nasze metryki klasyfikacyjne służą temu samemu porównaniu STL–MTL co na MEISD i nie są wynikiem leaderboardowym.

## Interpretacja analiz

Sparowane różnice MTL–STL odnoszą się do tego samego seedu, backbone'u, zbioru i zadania. Dla MAE korzyść odwraca znak. Raport zapisuje każdy seed, średnią, SD i opisowy 95% przedział t; pięć seedów nie obejmuje niepewności doboru danych ani błędów etykiet. Gradienty są liczone na stałej próbce train w trybie eval, bez zmiany wag, zapisanych gradientów czy generatora losowego. Wspólny kierunek gradientów, krzywe strat i routing ekspertów są wskazówkami o optymalizacji. Nie dowodzą semantycznego przepływu wiedzy między zadaniami.

Trening wymaga deterministycznych algorytmów PyTorch i ustawia `CUBLAS_WORKSPACE_CONFIG`. Jeśli docelowy GPU użyje operacji bez deterministycznej implementacji, run zakończy się błędem zamiast zapisać nieporównywalny wynik. Manifest odnotowuje tę politykę.

Pełne treningi nie zostały uruchomione. Krótkie przebiegi z losowym małym BERT sprawdzają wyłącznie działanie pipeline'u; ich metryk nie wolno traktować jako wyniku badawczego.
