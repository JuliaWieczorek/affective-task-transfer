# Affective Task Transfer

Replikacja i rozszerzenie badania z rozdziału 5: wspólne uczenie rozpoznawania wielu emocji, intensywności obecnych emocji i sentymentu. Projekt powstał przez migrację kodu z [`mtl-emotion-intensity-sentiment`](https://github.com/JuliaWieczorek/mtl-emotion-intensity-sentiment) na commicie `b5d8aaae5266229970b641f0b3b2e6e6da8607ea`. Oryginalne modele i ich pochodzenie są zapisane w `src/affective_task_transfer/models/inherited.py` oraz `docs/provenance.json`. Poprawki protokołu znajdują się w osobnych modułach.

## Cel badania

Porównujemy STL, wszystkie pary zadań oraz model trzyzadaniowy na MEISD. Zachowujemy historyczne rodziny architektur: hard sharing, soft sharing, adapters, MMoE, BERT-LSTM i cross-stitch, oraz cztery historyczne backbone'y: BERT uncased/cased, RoBERTa i XLM-R. Domyślny plan obejmuje pięć seedów (42, 52, 62, 72, 82). Zewnętrzny eksperyment na BRIGHTER English sprawdza tylko wspólne zadania: emocje i intensywność. Zbiór ten nie ma etykiet sentymentu; obecność emocji jest wyprowadzona z intensywności większej od zera.

Raporty obejmują accuracy, precision, recall, F1, macierze pomyłek, wyniki per klasa i zadanie, a także straty, normy i cosinusy gradientów oraz routing MMoE. Positive/negative transfer oznacza zmianę metryki względem sparowanego STL na tym samym zbiorze, backbone'ie, podziale i seedzie. Diagnostyki mogą wskazać mechanizm współdzielenia, ale same nie dowodzą przyczynowości.

Nie obejmujemy prognozowania, krzywych wielkości danych ani nowych metod augmentacji. Pełnych kosztownych eksperymentów nie uruchomiono; zaplanowane są po weryfikacji kodu i kosztu na docelowym komputerze. Stare repozytorium i wyniki nie są zmieniane.

## Dane i ważny wynik audytu

Źródła lokalne są w `data/source/meisd/`, archiwum i identyczna kopia jednego CSV w `data/archive/`. `data/` jest wyłączony z Git ze względu na wielkość i pochodzenie danych. Skróty SHA-256 są w `docs/data_audit.json` i w manifeście przygotowanego zbioru.

`multilabel_augmented_onehot_11222025.csv` ma 4219 wierszy, z czego 2608 ma `mode=llm` i puste `Utterances`. Dopasowany wiersz po wierszu `MEISD_balanced_expanded.csv` zawiera wszystkie 2608 tekstów w `augmented`. Adapter danych wymaga zgodności tekstu oryginalnego, rodzica, trybu, kolejności oraz etykiet w slotach przed połączeniem tekstu z etykietami one-hot. Sam plik one-hot nie wystarcza do odtworzenia augmentacji. W historycznym loaderze puste pole tekstowe powodowało użycie `original`, czyli ponowne podanie tekstu źródłowego. W konwersji poprawiono też 83 etykiety `positve` na `positive`. Dla 216 powtarzających się emocji ze sprzeczną intensywnością maskujemy tylko niejednoznaczną intensywność.

Jednostką MEISD jest historyczna *połówka dialogu*. Wszystkie 1611 oryginalnych tekstów da się odtworzyć z `MEISD_text.csv`, grupując wypowiedzi według `dialog_ids` i dzieląc w połowie. Podział 70/15/15 jest deterministyczny według dialogów, a identyczne teksty łączą grupy przed losowaniem. Dev i test zawierają tylko oryginały. Wszystkie augmentacje dziecka należą do grupy dialogu rodzica; do train trafiają tylko dzieci rodziców w train. Rekordy o tym samym tekście z rozbieżnymi etykietami są wykluczane, a identyczne kopie usuwane. To nowy podział, więc jego wyniki nie są bezpośrednią repliką historycznej tabeli walidacyjnej. Historyczne filtrowanie klas i odziedziczone etykiety pozostają ograniczeniami.

BRIGHTER English pobieramy z [oficjalnego zbioru Track B](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities) z przypiętym hashem rewizji i zachowaniem przynależności do oficjalnych train/dev/test. Audyt wyklucza 14 wierszy o identycznym tekście i sprzecznych etykietach oraz dwie zgodne kopie w train. [Opis zadania SemEval](https://github.com/emotion-analysis-project/semeval2025-task11) podaje klasy 0 (brak), 1 (niska), 2 (umiarkowana), 3 (wysoka) i pięć emocji dla języka angielskiego.

## Instalacja i przygotowanie

Python 3.11+; na Windows sprawdzono `torch==2.6.0+cpu` i `transformers==4.49.0`. Do eksperymentów GPU należy zainstalować kompatybilną wersję PyTorch dla docelowego CUDA.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\att.exe audit-data --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output docs/data_audit.json
.\.venv\Scripts\att.exe prepare-meisd --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output data/prepared/meisd-v2
.\.venv\Scripts\att.exe download-brighter --output data/source/brighter
.\.venv\Scripts\att.exe prepare-brighter --input data/source/brighter --output data/prepared/brighter-v1
```

`--expanded` jest opcjonalne tylko wtedy, gdy dopasowany plik leży obok CSV one-hot. Jeśli wiersze augmentowane nie mają tekstu i pliku nie ma, przygotowanie kończy się błędem.

## Weryfikacja i eksperymenty

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/smoke.py --dataset data/prepared/meisd-v2 --output outputs/smoke-meisd
.\.venv\Scripts\python.exe scripts/smoke.py --dataset data/prepared/brighter-v1 --output outputs/smoke-brighter --brighter
.\.venv\Scripts\att.exe make-matrix --dataset data/prepared/meisd-v2 --dataset-name meisd --output outputs/matrix-meisd-v2
.\.venv\Scripts\att.exe make-matrix --dataset data/prepared/brighter-v1 --dataset-name brighter --output outputs/matrix-brighter
```

Macierze są planami; samo ich utworzenie niczego nie trenuje. MEISD ma 240 konfiguracji, BRIGHTER 160. `att run-matrix --matrix outputs/matrix-meisd-v2/matrix.json` uruchamia pełny plan, a `--limit N` pierwsze N konfiguracji. Poszczególny run można wykonać przez `att train --config <plik.json>`. Po wybraniu checkpointu wyłącznie według dev wykonaj `att evaluate-matrix --matrix outputs/matrix-meisd-v2/matrix.json --split test` oraz analogicznie dla BRIGHTER. Pojedyncza ewaluacja: `att evaluate --dataset <przygotowany_zbior> --run <katalog_run> --split test`. Raport: `att report --runs outputs --output outputs/report --split test`. Tryb smoke używa małego losowego BERT, małych podzbiorów i jawnej flagi; jego wyniki nie są naukową oceną modeli.

Każdy run zapisuje konfigurację, wersje bibliotek, hash danych, seed, liczbę kroków, czas, checkpoint, predykcje i metryki. Raport agreguje wyniki po seedach oraz sparowane zmiany względem STL. Do pracy na drugim komputerze przenieś trzy pliki MEISD opisane w `docs/DATA_LAYOUT.md`; BRIGHTER pobiera CLI. Szczegóły audytu, pochodzenia i ograniczeń: `docs/IMPLEMENTATION_AUDIT.md`, `docs/provenance.json`; pierwotny plan pozostaje w `docs/INITIAL_PLAN_2026-09-28.md`.
