# Affective Task Transfer

> Historical planning snapshot (28 September 2026). The implemented Study II scope and commands are in `../README.md` and `STUDY_II_PROTOCOL.md`. Counts and proposed steps below are retained only as a record of the initial design.

## Plan projektu i badania do rozdziału 5

**Status:** plan przed implementacją, 28 września 2026 r. Opisane eksperymenty i komendy są projektowane; nie stanowią wykonanych badań ani działającego API.

**Proponowana nazwa repozytorium:** `affective-task-transfer`.

**Lokalizacja:** `D:/PycharmProjects/phd_research_code/affective-task-transfer`.

**Docelowy tytuł badania:** *Task-Level Knowledge Transfer for Joint Emotion, Intensity, and Sentiment Analysis: Reproducibility, Optimisation Diagnostics, and External Validation*.

Projekt będzie nową, samodzielną bazą kodu Python, wyprowadzoną z istniejącego projektu `mtl-emotion-intensity-sentiment`. Powstanie przez kontrolowaną migrację, refaktoryzację i rozszerzenie obecnych komponentów. Nie zakłada implementowania modeli od początku.

Na tym etapie powstał wyłącznie plan. Utworzenie lokalnego repozytorium Git i nowego repozytorium na GitHubie jest etapem realizacji; repozytorium zdalne nie zostało jeszcze utworzone. Przed publikacją należy ustalić konto docelowe i widoczność oraz przygotować czysty zestaw plików bez danych i checkpointów.

## 1. Cel i granice badania

Celem jest ustalenie, jak skład zadań oraz sposób współdzielenia parametrów wpływają na rozpoznawanie wielu emocji, przewidywanie intensywności każdej obecnej emocji i klasyfikację sentymentu.

Główne badanie pozostaje na MEISD, z dotychczasowymi danymi augmentowanymi, jeśli audyt potwierdzi możliwość ich poprawnego wykorzystania. ESConv pozostaje źródłem wzorców wykorzystanych w transformacji danych. Nie będzie zbiorem walidacyjnym ani testowym pełnego zadania trzyzadaniowego.

Nowy kod ma umożliwić:

- porównywanie STL, modeli dwuzadaniowych i trzyzadaniowych w zgodnych warunkach;
- ocenę stabilności wyników w kilku uruchomieniach;
- sprawdzenie mechanizmów soft sharing i diagnostykę optymalizacji;
- zewnętrzną replikację relacji emotion–intensity na niezależnym zbiorze;
- odtworzenie pochodzenia danych, konfiguracji i każdej tabeli wynikowej.

### Poza zakresem

- Krzywe wyników w funkcji ilości danych treningowych.
- Prognozowanie przyszłej intensywności, prefiksy dialogów i strategie wsparcia z rozdziału 6.
- Nowe porównanie generatorów i metod augmentacji z rozdziału 4.
- Pełny benchmark nowych modeli językowych i metod optymalizacji MTL.
- Duże badanie ponownej anotacji danych.
- Twierdzenia o skuteczności klinicznej lub generalizacji na wszystkie domeny.

Uwaga terminologiczna: procent dostępnej rozmowy w rozdziale 6 nie jest tym samym co procent dialogów w zbiorze treningowym. Żadnego z tych eksperymentów nie dodajemy do tego projektu jako głównego pytania badawczego.

## 2. Pytania badawcze

Identyfikatory poniżej są robocze; nie zastępują globalnej numeracji RQ w rozprawie.

| ID | Pytanie | Dowód |
|---|---|---|
| Q1 | Które połączenia zadań poprawiają lub pogarszają wyniki względem STL? | Sparowane porównania STL, par zadań i pełnego MTL, pięć seedów. |
| Q2 | Jak sposób współdzielenia parametrów zmienia te wyniki? | Hard sharing, soft sharing, adapters i MMoE; liczba parametrów i koszt obliczeń. |
| Q3 | Czy wspólna projekcja, regularyzacja encoderów i dynamika optymalizacji pomagają wyjaśnić zaobserwowane różnice? | Ablacje soft sharing, gradienty, straty zadań i routing ekspertów. |
| Q4 | Czy korzyści ze wspólnego uczenia emocji i intensywności powtarzają się na BRIGHTER English? | Zewnętrzna replikacja STL–MTL na oficjalnym podziale BRIGHTER. |

Nie zakładamy przewagi MTL. Brak poprawy, wyniki mieszane i pogorszenie konkretnego zadania są pełnoprawnymi rezultatami.

## 3. Pochodzenie kodu: migracja zamiast przepisywania

Repozytorium źródłowe: `../mtl-emotion-intensity-sentiment`.

Commit odczytany podczas przygotowania planu: `b5d8aaae5266229970b641f0b3b2e6e6da8607ea`. W źródłowym katalogu znajdują się również nieśledzone archiwa i kopie wyników; nie traktujemy ich automatycznie jako części tego commita. Przed migracją ustalamy, które artefakty odpowiadają wynikom rozprawy.

| Istniejący komponent | Sposób wykorzystania |
|---|---|
| `EMOTIA-ML/multi_emotion_sentiment_intensity_classifier.py` | Wydzielenie modeli, datasetu, funkcji strat i pętli treningowej do modułów. |
| `MultiTaskBERT`, `AdapterModule`, `MultiTaskBERTWithAdapters` | Zachowanie konstrukcji bazowych; jawne parametry konfiguracji i testy zgodności. |
| `MMOE_Core`, `MultiTaskMMOE` | Migracja istniejących ekspertów i bramek; dodanie opcjonalnego zapisu diagnostyki. |
| `SoftSharingModel` | Migracja faktycznie używanej definicji; rozdzielenie wspólnej projekcji i regularyzacji L2. |
| Modele STL, `FocalLoss`, `MultiTaskDataset` | Ponowne wykorzystanie po audycie poprawności i zgodności z protokołem. |
| `train_epoch`, `eval_epoch`, `run_pipeline` | Refaktoryzacja do wspólnego silnika obsługującego konfiguracje, checkpointy i metryki. |
| `EMOTIA-DA/one_hot_encoding.py` | Adaptacja konwersji; zachowanie identyfikatorów i braków anotacji. |
| `EMOTIA-DA/multilabel_augmenter.py` | Źródło wiedzy o pochodzeniu danych; generowanie nie jest domyślnie uruchamiane ponownie. |
| `external_validation/` | Wykorzystanie wzorca manifestów i porównań transferowych; adapter BRIGHTER wymaga odrębnego protokołu. |
| Skrypty analizy wyników | Ponowne wykorzystanie obliczeń dopiero po weryfikacji metryk i jednostki analizy. |

Nowy projekt będzie samodzielny: nie będzie importował modułów przez ścieżki do starego repozytorium. Zostaną zachowane informacje o autorstwie i pochodzeniu kodu. Przed publikacją należy sprawdzić licencję źródła i ustalić licencję nowego repozytorium.

### Oddzielenie refaktoryzacji od zmian metodologicznych

1. Zidentyfikować wykonywaną wersję kodu, historyczne konfiguracje i pliki wynikowe.
2. Zapisać mapę `stary plik/symbol → nowy plik/symbol` w `docs/PROVENANCE.md`.
3. Przenieść komponenty i sprawdzić zgodność forward pass przy tych samych wagach, wejściach i trybie ewaluacji.
4. Osobnymi zmianami wprowadzić poprawki metodologiczne i opisać ich wpływ.
5. Nowe wyniki raportować jako ponowne badanie z poprawionym protokołem. Nie zastępować po cichu historycznych tabel.

Historyczny backbone `bert-base-uncased` będzie modelem podstawowym. Pozostałe historyczne encodery mogą być obsługiwane przez konfigurację, ale pełny iloczyn encoderów, architektur i ablacji nie jest wymagany. Transfer kodu nie oznacza automatycznego ponownego użycia starych wytrenowanych checkpointów.

## 4. Audyt przed uruchomieniem eksperymentów

Wstępne czytanie kodu ujawniło kwestie do rozstrzygnięcia, a nie zakończony audyt historycznych wyników:

- Istnieją dwie definicje `SoftSharingModel`; należy ustalić faktycznie wykonywaną wersję.
- Opis rozprawy deklaruje maskowanie intensywności po emocjach obecnych w ground truth. Odczytana pętla treningowa liczy cross-entropy po wszystkich emocjach, bez jawnej maski. Trzeba prześledzić przygotowanie etykiet i wersję używaną do wyników.
- `parse_intensity` zamienia braki i wartości poniżej 1 na klasę 0, która odpowiada low po przesunięciu skali. W nowym protokole brak etykiety nie może stawać się niską intensywnością.
- Wagi strat występują w kilku miejscach. Domyślna konfiguracja nie jest wystarczającym dowodem konfiguracji historycznego eksperymentu.
- Wybór kolumny tekstowej musi być jawny; plik może zawierać zarówno oryginał, jak i augmentację.
- Należy sprawdzić liczenie multilabel F1, maskowanie ewaluacji, gradient accumulation, wybór checkpointu oraz progi emocji.

Wyniki audytu trafią do `docs/IMPLEMENTATION_AUDIT.md`. Nie uruchamiamy kosztownej macierzy eksperymentów przed rozstrzygnięciem błędów wpływających na cele uczenia i metryki.

## 5. Dane i ponowne wykorzystanie augmentacji

### MEISD: eksperyment główny

Schemat danych powinien przechowywać co najmniej: `sample_id`, `source_dialogue_id`, `parent_id`, `text`, `is_augmented`, `augmentation_method`, etykiety emocji, intensywności, sentymentu oraz maski dostępności anotacji.

Brak emocji, brak anotacji i najniższa intensywność są różnymi stanami. Słownik etykiet jest jawny i wersjonowany. Jeśli źródłowe wiersze są fragmentami jednego dialogu, podział obejmuje cały dialog, nie pojedynczy wiersz.

### Decyzja dotycząca istniejącego pliku

**Domyślnie próbujemy ponownie wykorzystać istniejące augmentacje.** Kod zapisuje `original` i `augmented`, co może umożliwić odzyskanie relacji źródłowych. Dostępność i kompletność tych pól w faktycznym końcowym CSV wymagają sprawdzenia.

Procedura:

1. Odnaleźć właściwy CSV, dane oryginalne i ewentualne logi generowania; zapisać ich hashe.
2. Odtworzyć źródłowe dialogi i rodziny augmentacji; zbadać duplikaty oraz niejednoznaczne dopasowania.
3. Ustalić i zapisać jeden grupowy podział oryginalnego MEISD: proponowane 70/15/15 train/dev/test, po sprawdzeniu liczebności klas.
4. W miarę możliwości zachować rozkład emocji, intensywności i sentymentu bez rozbijania grup. Algorytm i seed podziału ustalić przed treningiem.
5. Do train dołączyć tylko augmentacje źródeł przypisanych do train. Dev i test zawierają oryginalne teksty.
6. Sprawdzić, czy generowanie nie korzystało z przyszłych przykładów dev/test MEISD jako demonstracji lub materiału do uczenia transformacji.
7. Wagi klas, filtrowanie rzadkich kombinacji i decyzje o balansowaniu wyznaczać z train. Nie usuwać trudnych przykładów testowych tylko dlatego, że są rzadkie.

Jeżeli nie można potwierdzić pochodzenia części augmentacji, takie rekordy wykluczamy. Jeżeli odzyskanie pochodzenia lub niezależności jest niemożliwe dla całego pliku, przygotowujemy augmentacje wyłącznie z nowego train albo prowadzimy jawnie opisany wariant na danych oryginalnych. Nie uznajemy podobieństwa tekstowego za pewny dowód wspólnego źródła.

Można zachować dotychczasowe wzorce ESConv, jeżeli ESConv pozostaje zasobem pomocniczym, a nie niezależnym testem tego eksperymentu. Ponowne wykorzystanie tekstów nie eliminuje konieczności ponownego treningu przy nowym podziale.

## 6. Główna macierz eksperymentów

### Warunki wspólne

- Backbone podstawowy: `bert-base-uncased` z przypiętą wersją modelu/tokenizera.
- Pięć proponowanych seedów treningowych: `42, 52, 62, 72, 82`.
- Jeden zapisany podział danych, identyczny dla wszystkich warunków. Seed podziału oddzielony od seedów treningu.
- Ten sam zbiór treningowy, preprocessing, zakres informacji wejściowej i zasady oceny.
- Historyczne ustawienia jako punkt startowy: maksymalna długość 128, batch 16, AdamW, learning rate 2e-5, dropout 0.4, warmup 0.2. Ostateczne ustawienia zatwierdzić po audycie i pilotażu na train/dev.
- Ten sam budżet strojenia i jasno zdefiniowany wybór checkpointu. Kryterium wyboru nie może opierać się na nieporównywalnych surowych sumach strat różnych zestawów zadań.
- Proponowane wspólne kryterium MTL: średnia z dev macro-F1 aktywnych zadań, z ustalonym sposobem agregacji intensywności. STL używa metryki własnego zadania. Wybór zamknąć przed eksperymentami końcowymi.
- Próg emocji dobrać wyłącznie na dev, według tej samej procedury w każdym warunku. Jawnie zapisać politykę argmax fallback; na zbiorach dopuszczających brak emocji nie wymuszać dodatniej etykiety bez uzasadnienia.

### E1: skład zadań

Trzy modele STL: emotion, intensity, sentiment. Trzy pary soft sharing: emotion–intensity, emotion–sentiment, intensity–sentiment. Jeden pełny model soft sharing: emotion–intensity–sentiment.

Łącznie: **7 warunków × 5 seedów = 35 uruchomień**.

### E2: architektury współdzielenia

Pełne trzyzadaniowe MTL: hard sharing, soft sharing, adapters, MMoE. Wyniki soft sharing wykorzystujemy z E1.

Łącznie: **15 dodatkowych uruchomień**. Raportujemy całkowitą i trenowalną liczbę parametrów, liczbę encoderów, czas oraz szczytową pamięć GPU. Jednakowe hyperparametry nie gwarantują optymalnego dostrojenia każdej architektury; zakres strojenia ujawniamy.

### E3: ablacja soft sharing

Przy stałych zadaniach i liczbie encoderów porównujemy układ 2 × 2:

| Projekcja | Kara L2 między encoderami | Interpretacja |
|---|---|---|
| Osobna dla zadań | Wyłączona | Model bez tych dwóch dróg transferu. |
| Wspólna | Wyłączona | Wkład wspólnej projekcji. |
| Osobna dla zadań | Włączona | Wkład wiązania encoderów. |
| Wspólna | Włączona | Obecna konstrukcja soft sharing. |

Ostatni wariant wykorzystujemy z E1: **15 dodatkowych uruchomień**. Osobne projekcje nieznacznie zmieniają liczbę parametrów; podajemy tę różnicę. Wspólny wybór checkpointu może wpływać również na wariant bez transferu, więc nie utożsamiamy go automatycznie z trzema oddzielnie wybieranymi modelami STL.

Nie wykonujemy pełnego strojenia lambda jako nowego badania. Wybraną wartość oraz skalowanie sumy L2 dokumentujemy; sprawdzamy jej udział w gradientach.

## 7. Diagnostyka optymalizacji

Diagnostyka jest wykonywana w trakcie zaplanowanych treningów, na stałej próbce train i z ustaloną częstością. Pomiar nie może zmieniać gradientów używanych do aktualizacji ani sekwencji losowań głównego treningu. Czas pomiarów raportujemy osobno.

### Straty i gradienty

- Surowe i ważone straty każdego zadania, oddzielnie train/dev.
- Kara L2 jako osobny składnik, zamiast ukrywania jej w całkowitej stracie.
- Normy gradientów przed clippingiem i cosine similarity gradientów zadań na tych samych rzeczywiście współdzielonych parametrach.
- Częstość ujemnych cosine similarities i zmiany w kolejnych epokach.
- Liczba ważnych etykiet intensywności na batch, aby interpretować zmiany skali strat.

W hard sharing i MMoE badamy wspólny encoder oraz odpowiednie współdzielone bloki. W soft sharing gradienty różnych zadań porównujemy na `shared_fc`; na osobnych encoderach analizujemy relację gradientu zadania i gradientu regularyzacji. Nie przedstawiamy osobnych encoderów jako jednego wspólnego tensora parametrów.

### MMoE

Zapisujemy średnie wagi bramek per zadanie, entropię routingu, podobieństwo rozkładów między zadaniami, wykorzystanie ekspertów i zmiany w czasie. Dla tego modelu z miękką mieszaniną waga bramki nie oznacza wyłącznego przypisania przykładu do eksperta. Analizy klasowe wymagają raportowania liczebności.

Konflikt gradientów nie dowodzi przyczyny pogorszenia, a niska entropia nie jest sama w sobie dowodem zapadnięcia routingu. Diagnostyka służy testowaniu zgodności proponowanych wyjaśnień z obserwacjami. GradNorm/PCGrad pozostają rozszerzeniem opcjonalnym, poza podstawową macierzą.

## 8. E4: zewnętrzna replikacja na BRIGHTER

**Planowany zbiór:** BRIGHTER English, anotacje intensywności z SemEval-2025 Task 11 Track B. Wybór opiera się na wieloetykietowości i intensywnościach przypisanych do poszczególnych emocji, nie na spodziewanym wyniku modeli.

Angielski wariant obejmuje anger, fear, joy, sadness i surprise. Skala 0–3 rozróżnia brak emocji i trzy poziomy intensywności. Nie zapewnia niezależnej anotacji sentymentu potrzebnej do pełnej replikacji trzech zadań.

### Protokół podstawowy

- Zachować oficjalny train/dev/test i przypiąć wersję danych.
- Sprawdzić dostępność wszystkich anotacji oraz zgodność identyfikatorów, jeśli łączone są pliki kategorii i intensywności.
- Jeżeli obecność emocji jest wyprowadzana z intensywności > 0, ujawnić tę deterministyczną zależność. Nie przedstawiać obu etykiet jako niezależnych źródeł anotacji ani zakładać, że osobno agregowany Track A musi dawać identyczne etykiety.
- Używać tych samych podstawowych komponentów modeli, z nowymi głowami i natywną taksonomią BRIGHTER.
- Porównać STL emotion, STL intensity i soft-sharing MTL emotion–intensity; wszystkie od tego samego bazowego modelu językowego.
- Pięć seedów, identyczne dane i budżet strojenia; bez augmentacji BRIGHTER.
- Intensywność oceniać warunkowo dla obecnych emocji oraz w pełnym pipeline uwzględniającym wykrywanie obecności.
- Raportować wyniki osobno od MEISD. Nie łączyć ich w jeden wynik globalny.

Łącznie: **15 uruchomień**. Pytanie dotyczy replikacji efektu wspólnego uczenia dwóch zadań. Zbiór nie potwierdza skuteczności klinicznej, przewidywania przyszłych emocji ani generalizacji sentymentu.

### Opcjonalny eksperyment transferu reprezentacji

MEISD → BRIGHTER: bazowy encoder kontra pretrening MEISD STL kontra pretrening MEISD MTL, z jednakowym fine-tuningiem na BRIGHTER. To oddzielny eksperyment, poza podstawowym budżetem. Istniejący kod SemEval-2018 może dostarczyć elementów infrastruktury, ale nie zastępuje adaptera i protokołu BRIGHTER.

Nie nazywamy transferem bez adaptacji wyników uzyskanych po fine-tuningu. W razie pozostania przy samej E4 wnioskujemy o powtarzalności efektu MTL na drugim zbiorze.

## 9. Metryki i analiza niepewności

| Zadanie | Metryki |
|---|---|
| Emotion | Macro-F1 jako podstawowa, micro-F1 oraz precision/recall/F1 i support per emocja. |
| Intensity | Macro-F1 dla poziomów intensywności, agregowane według jawnej reguły po emocjach; MAE, accuracy, recall low/medium/high i macierze pomyłek. |
| Sentiment | Macro-F1 oraz metryki klasowe; weighted-F1 pomocniczo dla porównania z historycznymi tabelami. |
| Pełny pipeline | F1 par (emocja, poziom intensywności), z błędami wykrycia emocji liczonymi jako pominięcia lub fałszywe przewidywania. |

Warunkową intensywność liczymy tylko dla obecnych emocji z ważną anotacją intensywności. Brakujące etykiety wykluczamy za pomocą maski. Dla MAE i metryk porządkowych używamy oryginalnego porządku kategorii. Dla BRIGHTER można dodatkowo raportować oficjalną metrykę benchmarku po sprawdzeniu jego protokołu; nie zastępuje ona wspólnego raportu STL–MTL.

Zapisujemy wynik każdego seeda, średnią, SD i sparowane różnice MTL minus odpowiedni STL. Przedziały bootstrap dla predykcji liczymy z resamplingiem całych dialogów w MEISD, zachowując zależności między emocjami. Nie traktujemy etykiet jednego dialogu ani pięciu predykcji tego samego dialogu jako niezależnych obserwacji.

Zmienność między seedami i niepewność wynikająca ze skończonego testu są raportowane oddzielnie. Pięć seedów przy jednym podziale nie mierzy stabilności względem alternatywnych podziałów. Jeżeli dodajemy testy istotności, wcześniej definiujemy podstawowe kontrasty i rodzinę porównań oraz stosujemy korektę wielokrotnego testowania.

## 10. Planowana struktura repozytorium

```text
affective-task-transfer/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── experiments/          # E1–E4 i lista seedów
│   └── datasets/             # Schematy i mapowania etykiet
├── src/affective_task_transfer/
│   ├── cli.py
│   ├── config.py
│   ├── data/                # MEISD, BRIGHTER, maski, pochodzenie, podziały
│   ├── models/              # Wydzielone istniejące architektury
│   ├── losses.py
│   ├── training/            # Trainer, checkpointy, reproducibility
│   ├── diagnostics/         # Gradienty, straty, routing
│   └── evaluation/          # Metryki, porównania, tabele i wykresy
├── tests/                   # Testy poprawności i zgodności migracji
├── docs/
│   ├── PROVENANCE.md
│   ├── IMPLEMENTATION_AUDIT.md
│   ├── DATA_PROTOCOL.md
│   ├── EXPERIMENT_PROTOCOL.md
│   └── CHAPTER5_MAP.md
├── data/                    # Lokalne dane; ignorowane przez Git
├── manifests/               # Definicje eksperymentów; wrażliwe ID lokalnie
└── outputs/                 # Lokalne wyniki, logi, predykcje i checkpointy
```

Notebooki mogą służyć eksploracji. Wyniki do publikacji muszą powstawać przez moduły Python i wersjonowane konfiguracje, bez ręcznej edycji tabel.

Planowane polecenia CLI: `audit-data`, `prepare-data`, `train`, `run-experiments`, `evaluate`, `report`. Każde uruchomienie otrzyma manifest zawierający commit, hashe danych i splitu, konfigurację, seed, wersje bibliotek, sprzęt, checkpoint oraz ścieżki artefaktów.

## 11. Weryfikacja implementacji

Testy obejmą przede wszystkim ryzyka metodologiczne:

- rozłączność źródłowych dialogów i ich augmentacji między zbiorami;
- brak zamiany nieobecnej lub brakującej anotacji w low intensity;
- brak wpływu zamaskowanych pozycji na loss i metryki;
- obsługę batcha bez ważnych etykiet intensywności;
- poprawność multilabel F1 na ręcznie sprawdzonym małym przykładzie;
- poprawne wyłączanie zadań i elementów soft sharing;
- zgodność forward pass komponentów przeniesionych bez zmian;
- brak wpływu diagnostyki na aktualizacje treningu;
- zapis/odczyt checkpointu i kompletność manifestu.

Krótki smoke test ma potwierdzić działanie całego przepływu. Nie zastępuje eksperymentów badawczych ani nie jest dowodem odtworzenia historycznych wyników.

## 12. Etapy i warunki ukończenia

1. **Inwentaryzacja:** źródło kodu i wyników, plik augmentacji, mapowanie etykiet, audyt krytycznych rozbieżności.
2. **Migracja:** wydzielone komponenty z zachowanym pochodzeniem; testy zgodności i poprawek.
3. **Dane:** zapisany podział i manifest; decyzja o wykorzystaniu istniejących augmentacji; adapter BRIGHTER.
4. **Pilotaż:** wyłącznie train/dev; zamknięte konfiguracje, kryteria wyboru i główne kontrasty przed testem.
5. **E1–E3:** eksperymenty MEISD i diagnostyka.
6. **E4:** zewnętrzna replikacja na BRIGHTER.
7. **Raport:** metryki, niepewność, koszty i interpretacja wyników; mapowanie na podsekcje rozdziału 5.
8. **Repozytorium GitHub:** publikacja przygotowanego kodu, konfiguracji i dokumentacji; bez nieuprawnionego udostępniania korpusów, predykcji zawierających teksty lub checkpointów.

Podstawowy plan obejmuje **80 pełnych uruchomień**: 35 + 15 + 15 + 15, bez pilotażu i strojenia. Przed startem należy oszacować koszt na podstawie jednego krótkiego treningu. Opcjonalne backbone'y i transfer MEISD → BRIGHTER nie są wliczone. Jeżeli budżet wymaga redukcji, zmniejszamy liczbę dodatkowych konfiguracji przed oceną testową i dokumentujemy zmianę zakresu.

Badanie jest ukończone, gdy wyniki dają się odtworzyć z konfiguracji i manifestów, podstawowe porównania mają pięć uruchomień, test pozostaje niezależny od wyboru modeli, a każda tabela wskazuje dane, metrykę i jednostkę analizy. Nie uzależniamy ukończenia od uzyskania przewagi MTL.

## 13. Co projekt wniesie do rozprawy

| Kryterium | Zaplanowany wkład | Granica interpretacji |
|---|---|---|
| Credibility | Audyt danych i kodu, jawne pochodzenie, raportowanie zysków i strat. | Nowy protokół może dać inne wyniki niż historyczne tabele. |
| Rigour | Grupowy podział, zgodne porównania, powtórzenia, ablacje i niepewność. | Jeden podział nie obejmuje całej zmienności próbkowania danych. |
| Relevance | Analiza składów zadań i mechanizmów współdzielenia parametrów. | Bez dokładania forecastingu i badań wielkości danych. |
| Generalisability | Replikacja emotion–intensity na BRIGHTER. | Dwa zbiory nie dowodzą uniwersalności; BRIGHTER nie waliduje sentymentu ani kontekstu klinicznego. |
| Transferability | Jawny protokół i przenośna implementacja; opcjonalny test MEISD → BRIGHTER. | Przenośność kodu nie jest dowodem transferu wyuczonych reprezentacji. |

Planowane elementy rozdziału: opis poprawionego protokołu, tabela składów zadań, tabela architektur i kosztów, ablacja soft sharing, wykresy diagnostyczne, wyniki klasowe intensywności, zewnętrzna replikacja i ograniczenia. Dodatkowy dataset wzmacnia badanie, ale nie zastępuje poprawności głównego eksperymentu MEISD.

## 14. Źródła i zasoby

- Kod bazowy: `../mtl-emotion-intensity-sentiment/README.md` i `REPRODUCIBILITY.md`.
- [BRIGHTER — strona autorów](https://brighter-dataset.github.io/).
- [BRIGHTER — publikacja i opis anotacji](https://arxiv.org/html/2502.11926v2).
- [SemEval-2025 Task 11 — oficjalna dokumentacja](https://github.com/emotion-analysis-project/semeval2025-task11).
- [BRIGHTER emotion intensities — dane](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities).
- [SemEval-2018 Affect in Tweets](https://aclanthology.org/S18-1001/) — alternatywny, węższy eksperyment transferowy już opisany w źródłowym repozytorium.

Wybór BRIGHTER opiera się na dokumentacji sprawdzonej podczas planowania. Przed implementacją należy potwierdzić schemat pobranej wersji, warunki korzystania i hashe plików. To plan badania, nie deklaracja dostępności wyników ani zakończonej walidacji.
