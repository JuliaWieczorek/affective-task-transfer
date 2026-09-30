# Dane lokalne i przeniesienie na drugi komputer

Repozytorium GitHub zawiera kod, konfiguracje i audyt. `data/` jest lokalne i wyłączone z Git. Przenieś źródła MEISD do `data/source/meisd/` pod tymi nazwami:

| Plik | SHA-256 | Rola |
|---|---|---|
| `multilabel_augmented_onehot_11222025.csv` | `458add484602056689a64079385b47a01f3bdba1a9531a33ba23880b1934c4ad` | Etykiety po konwersji one-hot. |
| `MEISD_balanced_expanded.csv` | `20974b3232d2dd1e903d5b333e6267a826eb4845cb922d3a744e84f045e49019` | Teksty istniejących augmentacji i oryginalne sloty etykiet. |
| `MEISD_text.csv` | `30e5e212a2bf390af6d5fa01e8ac211072073a93b32c209917e276ca3957c749` | Surowe identyfikatory dialogów do bezpiecznego podziału. |

Kopia one-hot `(1).csv` jest identyczna i znajduje się lokalnie w `data/archive/`. `outputs22112025.zip` zachowano tam jako archiwum; nie jest potrzebne do uruchomienia pipeline'u. Katalogi `data/source/brighter/` i `data/prepared/` można odtworzyć komendami z głównego README. Pobranie BRIGHTER zapisuje hash rewizji, oficjalne pliki i pochodne CSV. Jeśli przygotowany zbiór już istnieje, CLI odmawia nadpisania manifestu: użyj nowej nazwy wersji.

Na nowym komputerze najpierw skopiuj trzy źródłowe pliki MEISD, sprawdź ich SHA-256 i uruchom `att audit-data`, `att prepare-meisd`, `att download-brighter`, `att prepare-brighter`. `att train` sprawdza hashe przygotowanych splitów, a `att evaluate` wymaga tego samego manifestu danych co trening. Nie kopiuj katalogu `outputs/` do repozytorium Git; do archiwizacji badań przenieś go osobno.
