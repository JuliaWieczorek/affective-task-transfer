# Przed pełnym uruchomieniem Study II — 2 października 2026, po południu

Wszystkie polecenia poniżej uruchamiaj w PowerShellu z katalogu
`D:\julixus\phd_research_code\affective-task-transfer`. Pełny podstawowy plan to
50 przebiegów MEISD i 30 przebiegów BRIGHTER, wykonywanych kolejno na jednej
karcie. Nie uruchamiaj jeszcze `run-matrix`, dopóki punkty „Warunek startu” nie
są spełnione.

## Stan potwierdzony 1 października

- [x] Gałąź `gpu-rocm-check`, commit `db74015`, zawiera zależność
  `torch>=2.9,<2.10` i konfigurację `configs/gpu-pilot-brighter.json`.
- [x] Python 3.12.2, PyTorch 2.9.1+rocm7.2.1; karta AMD Radeon RX 9070 jest
  widoczna dla PyTorch.
- [x] `pip check`: brak konfliktów. `pytest -q`: 27 testów zaliczonych.
- [x] BRIGHTER pobrany i przygotowany: 2753/115/2765 przykładów
  (train/dev/test), rewizja `08a5d61d3fc33036f97c8c76a61ff8d6f02f5157`.
- [x] Krótka próba BRIGHTER z prawdziwym BERT: 3 kroki na GPU, status
  `trained`, szczyt około 2,20 GiB VRAM. Wynik jakościowy tej próby nie jest
  wynikiem badawczym.
- [ ] Brakuje trzech plików źródłowych MEISD i przygotowanego zbioru MEISD.
- [ ] Nie zmierzono jeszcze pełnej partii 16 ani najbardziej wymagających
  architektur. Szacunek 40–100 godzin dla całości jest bardzo wstępny.

## 1. Zasoby komputera i środowisko

- [ ] Sprawdź wolne miejsce ponownie. 1 października było około 168,56 GiB na
  `D:` i tylko 3,74 GiB na `C:`. Na `D:` zachowaj co najmniej 100 GiB przed
  startem; na `C:` zwolnij bezpieczny zapas, najlepiej co najmniej 10–15 GiB.
  Pojedynczy zapisany checkpoint BERT miał 419 MiB; pełna macierz może zająć
  około 45–60 GiB na checkpointy, dodatkowo cache i wyniki.
- [ ] W **każdym nowym terminalu**, przed uruchomieniem Pythona, skieruj cache
  Hugging Face i pliki tymczasowe na `D:`:

  ```powershell
  Set-Location D:\julixus\phd_research_code\affective-task-transfer
  $env:HF_HOME = 'D:\julixus\phd_research_code\.hf-cache'
  $env:TEMP = 'D:\julixus\phd_research_code\.tmp'
  $env:TMP = $env:TEMP
  New-Item -ItemType Directory -Force $env:HF_HOME,$env:TEMP | Out-Null
  [System.IO.DriveInfo]::GetDrives() | Where-Object Name -in 'C:\','D:\' |
    Select-Object Name,@{N='FreeGiB';E={[math]::Round($_.AvailableFreeSpace/1GB,1)}}
  ```

  To nie przenosi istniejącego cache z `C:`; nowy cache na `D:` może ponownie
  pobrać BERT. Nie usuwaj starego cache przed potwierdzeniem nowych plików.
- [ ] Zamknij zbędne programy obciążające RAM/VRAM. 1 października komputer miał
  15,92 GiB RAM (około 5,11 GiB dostępne) i 15,92 GiB VRAM.
- [ ] Potwierdź gałąź, biblioteki i GPU. Jeżeli zmieniły się pakiety lub kod,
  ponów testy:

  ```powershell
  git status --short --branch
  .\.venv\Scripts\python.exe -m pip check
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

  Oczekiwane: gałąź `gpu-rocm-check`, brak konfliktów, 27 zaliczonych testów,
  PyTorch z `+rocm7.2.1`, `True` i `AMD Radeon RX 9070`.

## 2. Dane MEISD i kontrola przygotowania

- [ ] Umieść poniższe pliki w `data/source/meisd/` i porównaj ich SHA-256 z
  wartościami w [`DATA_LAYOUT.md`](DATA_LAYOUT.md):

  | Plik | Początek oczekiwanego SHA-256 |
  | --- | --- |
  | `multilabel_augmented_onehot_11222025.csv` | `458add484602` |
  | `MEISD_balanced_expanded.csv` | `20974b3232d2` |
  | `MEISD_text.csv` | `30e5e212a2bf` |

  ```powershell
  Get-FileHash data/source/meisd/*.csv -Algorithm SHA256 | Select-Object Path,Hash
  ```

- [ ] Wykonaj audyt i przygotowanie tylko raz do nowego katalogu:

  ```powershell
  .\.venv\Scripts\att.exe audit-data --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output outputs/preflight-meisd-audit.json
  .\.venv\Scripts\att.exe prepare-meisd --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output data/prepared/meisd-study2-v1
  ```

- [ ] Otwórz `data/prepared/meisd-study2-v1/manifest.json`: oczekiwane
  train/dev/test to **1378/118/118**, a `audit.prepared_segment` ma wartość
  `start`. Potwierdź też istniejący manifest BRIGHTER: **2753/115/2765**.
  Przygotowanie odmawia nadpisania istniejącego manifestu; nie usuwaj danych
  tylko po to, żeby ponowić polecenie.

## 3. Krótkie pomiary przed dużą macierzą

- [ ] Na obu zbiorach zmierz osobne, krótkie przebiegi z prawdziwym
  `bert-base-uncased` dla `hard_sharing`, `soft_sharing`, `adapters` i `mmoe`.
  Użyj docelowej partii `batch_size=16`, `device="cuda"`, włączonych
  `diagnostics`, a do ograniczenia czasu ustaw `smoke=true`, `max_steps`
  około 20–50 i osobne katalogi `outputs/preflight-*`. BRIGHTER ma zadania
  emotion/intensity; MEISD ma sentiment/emotion/intensity. Nie używaj
  `scripts/smoke.py` do pomiaru GPU — wymusza CPU.
- [ ] Z każdego `manifest.json` zapisz `seconds_per_optimizer_step`,
  `diagnostic_seconds`, `peak_cuda_bytes`, `elapsed_seconds`, liczbę kroków
  i wielkość `best_model.pt`. Sprawdź, czy żaden przebieg nie kończy się błędem
  braku VRAM. Najważniejsza próba pamięci to MEISD `soft_sharing` z trzema
  zadaniami i partią 16.
- [ ] Przelicz prognozę 80 przebiegów z pomiarów. Uwzględnij 3–6 epok,
  walidację, diagnostykę, zapis checkpointów i zapas na ponowienia. Jeśli
  `batch_size=16` nie mieści się w VRAM, ustal nową partię i akumulację
  gradientów **przed** wygenerowaniem oficjalnej macierzy; zapisz zmianę
  protokołu i ponów oszacowanie czasu.

## 4. Warunek startu oficjalnej macierzy

- [ ] Poprzednie punkty są odhaczone, nie ma konfliktów pakietów ani błędów
  testów, oba manifesty danych mają oczekiwane liczności, a próby GPU mieszczą
  się w pamięci.
- [ ] Wygeneruj **nowe**, osobne macierze z katalogu głównego projektu:

  ```powershell
  .\.venv\Scripts\att.exe make-matrix --dataset data/prepared/meisd-study2-v1 --dataset-name meisd --output outputs/matrix-meisd-study2-v2
  .\.venv\Scripts\att.exe make-matrix --dataset data/prepared/brighter-study2-v1 --dataset-name brighter --output outputs/matrix-brighter-study2-v2
  ```

  Oczekiwane są 50 i 30 konfiguracji. Nie mieszaj pilotażowych katalogów
  `outputs/preflight-*` z macierzami. `make-matrix` nie nadpisuje niepustego
  katalogu; `run-matrix` nie wznawia niekompletnego przebiegu bez ręcznej
  decyzji. Zachowaj pliki macierzy, konfiguracje, manifesty oraz wynik `git
  rev-parse HEAD` razem z rezultatami.
- [ ] Dopiero po zaakceptowaniu czasu i zasobów uruchom `run-matrix` według
  [`RUN_HANDOFF.md`](RUN_HANDOFF.md). Testowy split oceniaj dopiero po
  zakończeniu i przeglądzie całej zaplanowanej macierzy.

Ta lista dotyczy podstawowej macierzy Study II. Dodatkowe backbone'y,
architektury historyczne i ablacje wymagają osobnego planu czasu i miejsca.
