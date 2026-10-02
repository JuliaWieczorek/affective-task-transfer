# Study II — kontrola przed pełnym uruchomieniem (2 października 2026)

Wszystkie bloki uruchamiaj w **tym samym terminalu PowerShell** z katalogu
`D:\julixus\phd_research_code\affective-task-transfer`. Podstawowa macierz
to 50 przebiegów MEISD i 30 BRIGHTER, kolejno na jednej karcie. Próby poniżej
mają `smoke=true` i korzystają tylko z train/dev. **Nie uruchamiaj
`run-matrix` ani oceny testowej przed zielonym światłem.** Nie kasuj
niekompletnych katalogów: zachowaj je do diagnozy i dla ponowienia
wybierz nową nazwę.

## Stan potwierdzony 1 października

- Pełna jedna epoka BRIGHTER `hard_sharing` z prawdziwym BERT, partią 16
  i historycznym limitem **128 tokenów**:
  172 kroki, **89,2 s** (z walidacją, diagnostyką i zapisem checkpointu),
  **3,86 GiB** szczytowo przydzielonej pamięci GPU, **1,7 s**
  diagnostyki. Checkpoint 419 MiB daje się odczytać, jego hash jest zgodny
  z manifestem, a metryki dev odtwarzają się z predykcji. Testu nie oceniano.
- Emotion macro-F1 po jednej epoce: **0,5796** wobec baseline **0,1401**.
  Intensity macro-F1 po emocjach: **0,2392**, równe baseline. Wszystkie
  aktywne intensywności przewidziano jako `low`; recall `medium` i `high`
  wyniósł zero. To wymaga kontroli po kilku epokach, lecz nie jest jeszcze
  końcową oceną modelu.
- Lokalne `pytest -q`: 27 testów zaliczonych; `pip check`: bez konfliktów.
  Na komputerze GPU działał Python 3.12.2, PyTorch 2.9.1+rocm7.2.1
  i AMD Radeon RX 9070. Zmiany z `gpu-rocm-check` są już w `main` (PR #1).
- **Nowy protokół używa 192 tokenów.** Powyższe 89,2 s oraz wcześniejsze
  oszacowania czasu dla 128 tokenów są tylko punktem odniesienia;
  harmonogram trzeba wyliczyć na nowo z prób przy 192 tokenach.
- Końcowe checkpointy pełnej macierzy mogą zająć około **45 GiB** (55 runów z
  jednym, 20 z dwoma i 5 z trzema encoderami). Planuj **45–60 GiB**
  plus cache, predykcje, archiwum i przejściowy checkpoint umożliwiający
  wznowienie bieżącego runu. Na komputerze GPU brakowało jeszcze
  trzech plików źródłowych MEISD.

## 1. Komputer, kod, środowisko

1 października wolne miejsce wynosiło ok. 168,56 GiB na `D:` i 3,74 GiB
na `C:`. Przed startem wymagaj **co najmniej 100 GiB na `D:` i 10 GiB
na `C:`**. Skieruj cache Hugging Face i pliki tymczasowe na `D:` w każdym
nowym terminalu, zanim uruchomisz Pythona. Nie usuwaj starego cache
ani danych badawczych bez sprawdzenia nowych kopii.

```powershell
$project = 'D:\julixus\phd_research_code\affective-task-transfer'
Set-Location $project
$env:HF_HOME = 'D:\julixus\phd_research_code\.hf-cache'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TEMP = 'D:\julixus\phd_research_code\.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME,$env:HF_HUB_CACHE,$env:TEMP | Out-Null
Get-PSDrive -Name C,D | Select-Object Name,@{N='FreeGiB';E={[math]::Round($_.Free/1GB,1)}}
if ((Get-PSDrive -Name D).Free -lt 100GB) { throw 'Za mało miejsca na D:' }
if ((Get-PSDrive -Name C).Free -lt 10GB) { throw 'Za mało miejsca na C:' }
```

Zamknij zbędne programy obciążające RAM i VRAM. Sprawdź niezapisane
zmiany **przed** przełączeniem na `main`. Nie używaj `reset --hard`.

```powershell
git status --short --branch
if (git status --porcelain) { throw 'Najpierw sprawdź lokalne zmiany' }
git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'git fetch nie powiódł się' }
git switch main
if ($LASTEXITCODE -ne 0) { throw 'Nie można przełączyć na main' }
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Nie można pobrać main' }
git lfs pull
if ($LASTEXITCODE -ne 0) { throw 'Nie pobrano Git LFS' }
git rev-parse HEAD
git status --short --branch
```

`torch.cuda` jest interfejsem PyTorch także na ROCm. Zapisz osobno
wersję HIP i nazwę GPU: historyczny manifest ma `cuda: null`.

```powershell
.\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Konflikt pakietów' }
.\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Testy nie przeszły' }
@'
import torch
assert torch.cuda.is_available(), "PyTorch nie widzi GPU"
p = torch.cuda.get_device_properties(0)
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", p.name)
print("VRAM GiB:", round(p.total_memory / 2**30, 2))
assert torch.version.hip, "To nie jest build ROCm"
'@ | .\.venv\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Kontrola ROCm/GPU nie przeszła' }
```

Zielone: AMD Radeon RX 9070, sprawdzony build ROCm, wszystkie testy
zielone i brak konfliktów. Nowy manifest runu powinien zapisać
`environment.hip`, nazwę GPU i pamięć. Po zmianie kodu lub pakietów
ponów pomiary.

## 2. Dane MEISD i BRIGHTER

Umieść poniższe pliki w `data/source/meisd/`. Jeśli znajdują się na
nośniku, zmień tylko `$transferDir` na rzeczywistą ścieżkę. Blok
kopiuje wyłącznie brakujące pliki i porównuje pełne SHA-256.

```powershell
$transferDir = 'E:\MEISD-transfer' # ZMIEŃ na katalog z trzema plikami
$sourceDir = 'data/source/meisd'
New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
$expected = @{
    'multilabel_augmented_onehot_11222025.csv' = '458add484602056689a64079385b47a01f3bdba1a9531a33ba23880b1934c4ad'
    'MEISD_balanced_expanded.csv' = '20974b3232d2dd1e903d5b333e6267a826eb4845cb922d3a744e84f045e49019'
    'MEISD_text.csv' = '30e5e212a2bf390af6d5fa01e8ac211072073a93b32c209917e276ca3957c749'
}
foreach ($name in $expected.Keys) {
    $target = Join-Path $sourceDir $name
    if (-not (Test-Path -LiteralPath $target)) {
        $from = Join-Path $transferDir $name
        if (-not (Test-Path -LiteralPath $from)) { throw "Brak pliku: $from" }
        Copy-Item -LiteralPath $from -Destination $target
    }
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$name]) { throw "Niezgodny SHA-256: $name" }
    Write-Host "$name OK"
}
```

Przygotowanie odmawia nadpisania istniejącego manifestu. Uruchom audyt
i przygotowanie tylko dla brakujących wyników.

```powershell
$auditPath = 'outputs/preflight-meisd-audit-2026-10-02.json'
if (-not (Test-Path -LiteralPath $auditPath)) {
    & .\.venv\Scripts\att.exe audit-data --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output $auditPath
    if ($LASTEXITCODE -ne 0) { throw 'Audyt MEISD nie przeszedł' }
}
if (-not (Test-Path 'data/prepared/meisd-study2-v1/manifest.json')) {
    & .\.venv\Scripts\att.exe prepare-meisd --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output data/prepared/meisd-study2-v1
    if ($LASTEXITCODE -ne 0) { throw 'Przygotowanie MEISD nie przeszło' }
}
if (-not (Test-Path 'data/source/brighter/download_manifest.json')) {
    & .\.venv\Scripts\att.exe download-brighter --output data/source/brighter
    if ($LASTEXITCODE -ne 0) { throw 'Pobranie BRIGHTER nie przeszło' }
}
if (-not (Test-Path 'data/prepared/brighter-study2-v1/manifest.json')) {
    & .\.venv\Scripts\att.exe prepare-brighter --input data/source/brighter --output data/prepared/brighter-study2-v1
    if ($LASTEXITCODE -ne 0) { throw 'Przygotowanie BRIGHTER nie przeszło' }
}
```

Sprawdź hashe splitów przez `load_dataset`, liczności, `start`,
brak augmentacji w dev/test i brak przenikania grup lub identycznych
tekstów. BRIGHTER musi mieć przypiętą rewizję
`08a5d61d3fc33036f97c8c76a61ff8d6f02f5157`.

```powershell
@'
from itertools import combinations
from affective_task_transfer.data import load_dataset, normalize

expected = {"meisd-study2-v1": (1378,118,118),
            "brighter-study2-v1": (2753,115,2765)}
for name, counts in expected.items():
    splits, manifest = load_dataset("data/prepared/" + name)
    assert tuple(len(splits[s]) for s in ("train","dev","test")) == counts
    groups = {s: {r["group_id"] for r in splits[s]} for s in splits}
    texts = {s: {normalize(r["text"]) for r in splits[s]} for s in splits}
    for a,b in combinations(("train","dev","test"),2):
        assert groups[a].isdisjoint(groups[b]), (name,a,b,"groups")
        assert texts[a].isdisjoint(texts[b]), (name,a,b,"text")
    if name.startswith("meisd"):
        assert manifest["audit"]["prepared_segment"] == "start"
        assert all(r["segment"] == "start" for rows in splits.values() for r in rows)
        assert not any(r["is_augmented"] for s in ("dev","test") for r in splits[s])
    else:
        assert manifest["audit"]["revision"] == "08a5d61d3fc33036f97c8c76a61ff8d6f02f5157"
        assert manifest["audit"]["test_labels_used_for_cleaning"] is False
    print(name, counts, "OK")
'@ | .\.venv\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Kontrola danych nie przeszła' }
```

Automatyczne kontrole przygotowania danych nie potwierdzają semantycznej
poprawności odziedziczonych etykiet. Zachowaj to jako ograniczenie badania;
ręczny przegląd próbek nie jest częścią protokołu ani warunkiem startu.

## 3. Pełnoepokowe próby GPU przed macierzą

Istniejąca próba BRIGHTER `hard_sharing` jest poprawną próbą przy 128
tokenach. Sprawdź jej archiwum, ale **nie używaj jej do oszacowania
nowego protokołu 192-tokenowego**:

```powershell
$oldRun = 'outputs/preflight-brighter-hard-sharing-2026-10-01'
$oldManifest = Get-Content (Join-Path $oldRun 'manifest.json') -Raw | ConvertFrom-Json
$oldHash = (Get-FileHash (Join-Path $oldRun 'best_model.pt') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($oldManifest.status -ne 'trained' -or $oldManifest.optimizer_steps -ne 172 -or $oldHash -ne $oldManifest.checkpoint_sha256) { throw 'Archiwalna próba jest niekompletna' }
```

Uruchom **11 prób po jednej pełnej epoce**. Każda używa
całego train/dev, prawdziwego BERT, partii 16, limitu 192 tokenów
i osobnego katalogu.
Najpierw idzie najtrudniejszy pamięciowo MEISD `soft_sharing` z trzema
zadaniami. `matched_stl` i dwuzadaniowy soft sharing są potrzebne do
oszacowania całej macierzy. `scripts/smoke.py` wymusza CPU i nie nadaje
się do pomiaru GPU. Jeśli `$pilotRoot` istnieje, wybierz nowy sufiks;
nie kasuj starego katalogu.

```powershell
$pilotRoot = 'outputs/preflight-study2-2026-10-02'
if (Test-Path -LiteralPath $pilotRoot) { throw "Katalog już istnieje: $pilotRoot" }
New-Item -ItemType Directory -Force -Path (Join-Path $pilotRoot 'configs') | Out-Null
$configDir = (Resolve-Path (Join-Path $pilotRoot 'configs')).Path
$datasets = @{ meisd='data/prepared/meisd-study2-v1'; brighter='data/prepared/brighter-study2-v1' }
$cases = @(
    @{ name='meisd-soft-triple'; dataset='meisd'; arch='soft_sharing'; tasks=@('sentiment','emotion','intensity'); diagnostics=$true }
    @{ name='meisd-soft-pair'; dataset='meisd'; arch='soft_sharing'; tasks=@('emotion','intensity'); diagnostics=$true }
    @{ name='meisd-stl-emotion'; dataset='meisd'; arch='matched_stl'; tasks=@('emotion'); diagnostics=$false }
    @{ name='meisd-hard'; dataset='meisd'; arch='hard_sharing'; tasks=@('sentiment','emotion','intensity'); diagnostics=$true }
    @{ name='meisd-adapters'; dataset='meisd'; arch='adapters'; tasks=@('sentiment','emotion','intensity'); diagnostics=$true }
    @{ name='meisd-mmoe'; dataset='meisd'; arch='mmoe'; tasks=@('sentiment','emotion','intensity'); diagnostics=$true }
    @{ name='brighter-soft'; dataset='brighter'; arch='soft_sharing'; tasks=@('emotion','intensity'); diagnostics=$true }
    @{ name='brighter-hard'; dataset='brighter'; arch='hard_sharing'; tasks=@('emotion','intensity'); diagnostics=$true }
    @{ name='brighter-stl-intensity'; dataset='brighter'; arch='matched_stl'; tasks=@('intensity'); diagnostics=$false }
    @{ name='brighter-adapters'; dataset='brighter'; arch='adapters'; tasks=@('emotion','intensity'); diagnostics=$true }
    @{ name='brighter-mmoe'; dataset='brighter'; arch='mmoe'; tasks=@('emotion','intensity'); diagnostics=$true }
)
foreach ($case in $cases) {
    $cfg = @{ backbone='bert-base-uncased'; architecture=$case.arch; tasks=$case.tasks;
        seed=42; dataset=$datasets[$case.dataset]; output=(Join-Path $pilotRoot $case.name);
        device='cuda'; smoke=$true; epochs=1; batch_size=16; max_length=192;
        diagnostics=$case.diagnostics; diagnostic_batch_size=32 }
    [System.IO.File]::WriteAllText((Join-Path $configDir ($case.name + '.json')),
        ($cfg | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
}
foreach ($case in $cases) {
    & .\.venv\Scripts\att.exe train --config (Join-Path $configDir ($case.name + '.json'))
    if ($LASTEXITCODE -ne 0) { throw "Próba nie przeszła: $($case.name)" }
}
```

Każdy manifest musi mieć `status=trained`, `smoke=true`,
`batch_size=16`, `peak_cuda_bytes` i skończone straty. Oczekuj
87 kroków MEISD i 172 BRIGHTER (ostatni, jednoelementowy batch BRIGHTER
jest łączony z poprzednim). Modele wielozadaniowe muszą zapisać
diagnostykę; MMoE dodatkowo `routing`. Przy OOM, NaN/Inf lub
niekompletnym checkpointcie **wstrzymaj macierz**.

Po wszystkich próbach sprawdź zawartość historii i diagnostyk, także
sumę wag bramek MMoE:

```powershell
@'
import json, math
from pathlib import Path

root = Path("outputs/preflight-study2-2026-10-02")
cases = {
    "meisd-soft-triple": ("meisd", True),
    "meisd-soft-pair": ("meisd", True),
    "meisd-stl-emotion": ("meisd", False),
    "meisd-hard": ("meisd", True),
    "meisd-adapters": ("meisd", True),
    "meisd-mmoe": ("meisd", True),
    "brighter-soft": ("brighter", True),
    "brighter-hard": ("brighter", True),
    "brighter-stl-intensity": ("brighter", False),
    "brighter-adapters": ("brighter", True),
    "brighter-mmoe": ("brighter", True),
}
pilots = [(root / name, dataset, diagnostic) for name, (dataset, diagnostic) in cases.items()]

def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)

for run, dataset, expected_diagnostics in pilots:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected_steps = 87 if dataset == "meisd" else 172
    assert manifest["status"] == "trained" and manifest["smoke"], run
    assert manifest["config"]["max_length"] == 192, (run, "max length")
    assert manifest["optimizer_steps"] == expected_steps, (run, "steps")
    history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(history) == 1 and finite_tree(history[0]), (run, "history")
    assert history[0]["optimizer_steps"] == expected_steps, (run, "history steps")
    assert all(history[0]["train_losses"][task] is not None
               for task in manifest["config"]["tasks"]), (run, "train losses")
    assert math.isfinite(history[0]["selection_score"]), (run, "dev score")
    if expected_diagnostics:
        diagnostics = [json.loads(line) for line in (run / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(diagnostics) == 1 and finite_tree(diagnostics[0]), (run, "diagnostics")
        assert diagnostics[0]["losses"] and diagnostics[0]["gradient_norms"], (run, "gradients")
        if "mmoe" in run.name:
            routing = diagnostics[0]["routing"]
            assert set(routing) == set(manifest["config"]["tasks"]), (run, "routing tasks")
            for task, gate in routing.items():
                assert abs(sum(gate["mean_weights"]) - 1) < 0.001, (run, task, "routing weights")
    print(run, "OK")
'@ | .\.venv\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Historia lub diagnostyka próby są niekompletne' }
```

## 4. Czy intensywność zaczyna się uczyć?

Jedna epoka dała tylko `low`. Uruchom dwa osobne przebiegi BRIGHTER
do maksymalnie 6 epok: `hard_sharing` oraz `matched_stl` intensywności.
Wczesne zatrzymanie może skończyć po 3–6 epokach. Oceniaj tylko dev.
Te katalogi są `smoke=true` i nie należą do wyników Study II.

```powershell
$learningRoot = 'outputs/preflight-study2-learning-2026-10-02'
if (Test-Path -LiteralPath $learningRoot) { throw "Katalog już istnieje: $learningRoot" }
New-Item -ItemType Directory -Force -Path (Join-Path $learningRoot 'configs') | Out-Null
$learningDir = (Resolve-Path (Join-Path $learningRoot 'configs')).Path
$learningCases = @(
    @{ name='brighter-hard'; arch='hard_sharing'; tasks=@('emotion','intensity'); diagnostics=$true }
    @{ name='brighter-stl-intensity'; arch='matched_stl'; tasks=@('intensity'); diagnostics=$false }
)
foreach ($case in $learningCases) {
    $cfg = @{ backbone='bert-base-uncased'; architecture=$case.arch; tasks=$case.tasks;
        seed=42; dataset='data/prepared/brighter-study2-v1';
        output=(Join-Path $learningRoot $case.name); device='cuda'; smoke=$true;
        epochs=6; batch_size=16; max_length=192; diagnostics=$case.diagnostics; diagnostic_batch_size=32 }
    [System.IO.File]::WriteAllText((Join-Path $learningDir ($case.name + '.json')),
        ($cfg | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    & .\.venv\Scripts\att.exe train --config (Join-Path $learningDir ($case.name + '.json'))
    if ($LASTEXITCODE -ne 0) { throw "Trening nie przeszedł: $($case.name)" }
    & .\.venv\Scripts\att.exe evaluate --dataset data/prepared/brighter-study2-v1 --run (Join-Path $learningRoot $case.name) --split dev
    if ($LASTEXITCODE -ne 0) { throw "Ocena dev nie przeszła: $($case.name)" }
}
foreach ($case in $learningCases) {
    $run = Join-Path $learningRoot $case.name
    Get-Content (Join-Path $run 'history.jsonl')
    $r = Get-Content (Join-Path $run 'dev_results.json') -Raw | ConvertFrom-Json
    $cm = $r.metrics.intensity.confusion_matrix
    [pscustomobject]@{ Run=$case.name;
        IntensityMacroF1=$r.metrics.intensity.emotion_macro_f1;
        MajorityMacroF1=$r.majority_baseline.intensity.emotion_macro_f1;
        PredictedMedium=(@($cm | ForEach-Object { $_[1] } | Measure-Object -Sum).Sum);
        PredictedHigh=(@($cm | ForEach-Object { $_[2] } | Measure-Object -Sum).Sum) }
}
```

Jeśli **oba** kilkuepokowe modele nadal przewidują tylko `low`,
wstrzymaj pełny start do kontroli rozkładu klas, maskowania i straty. Słaby
F1 sam w sobie nie dowodzi błędu: po jego wykluczeniu może być
prawdziwym negatywnym wynikiem. Nie dobieraj hiperparametrów po
zajrzeniu do testu.

## 5. Pomiary i prognoza 80 runów

Użyj `elapsed_seconds` z pełnych jednoepokowych prób. Pole
`seconds_per_optimizer_step` obejmuje także walidację i checkpoint
podzielone przez liczbę kroków; nie jest czystym czasem kroku.
Poniższa formuła liczy 5 seedów, 3 STL + 3 pary + 4 architektury MEISD
oraz 2 STL + 4 architektury BRIGHTER. Próbę jednego STL i jednej pary
traktuje jako reprezentatywną dla pokrewnych warunków.

```powershell
$pilotRoot = 'outputs/preflight-study2-2026-10-02'
$paths = @{
    meisd_stl=(Join-Path $pilotRoot 'meisd-stl-emotion')
    meisd_pair=(Join-Path $pilotRoot 'meisd-soft-pair')
    meisd_hard=(Join-Path $pilotRoot 'meisd-hard')
    meisd_triple=(Join-Path $pilotRoot 'meisd-soft-triple')
    meisd_adapters=(Join-Path $pilotRoot 'meisd-adapters')
    meisd_mmoe=(Join-Path $pilotRoot 'meisd-mmoe')
    brighter_stl=(Join-Path $pilotRoot 'brighter-stl-intensity')
    brighter_hard=(Join-Path $pilotRoot 'brighter-hard')
    brighter_soft=(Join-Path $pilotRoot 'brighter-soft')
    brighter_adapters=(Join-Path $pilotRoot 'brighter-adapters')
    brighter_mmoe=(Join-Path $pilotRoot 'brighter-mmoe')
}
$times = @{}
$pilotRows = foreach ($key in $paths.Keys) {
    $run = $paths[$key]
    $m = Get-Content (Join-Path $run 'manifest.json') -Raw | ConvertFrom-Json
    if ($m.status -ne 'trained' -or -not $m.smoke -or $m.config.batch_size -ne 16 -or
        $m.config.max_length -ne 192 -or -not $m.environment.hip -or -not $m.environment.gpu.name -or
        $null -eq $m.peak_cuda_bytes -or $m.resumable_checkpoint_bytes -le 0 -or $m.elapsed_seconds -le 0) {
        throw "Niekompletna próba: $key"
    }
    $checkpoint = Join-Path $run 'best_model.pt'
    $hash = (Get-FileHash $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $m.checkpoint_sha256) { throw "Niezgodny checkpoint: $key" }
    if ($m.config.diagnostics -and -not (Test-Path (Join-Path $run 'diagnostics.jsonl'))) {
        throw "Brak diagnostyki: $key"
    }
    $times[$key] = [double]$m.elapsed_seconds
    [pscustomobject]@{ Case=$key; Steps=$m.optimizer_steps;
        EpochSeconds=[math]::Round($m.elapsed_seconds,1);
        DiagnosticsSeconds=[math]::Round($m.diagnostic_seconds,1);
        PeakVRAMGiB=[math]::Round($m.peak_cuda_bytes/1GB,2);
        ResumeCheckpointGiB=[math]::Round($m.resumable_checkpoint_bytes/1GB,2);
        CheckpointMiB=[math]::Round((Get-Item $checkpoint).Length/1MB,1) }
}
$pilotRows | Sort-Object Case | Format-Table -AutoSize
$pilotRows | Export-Csv (Join-Path $pilotRoot 'timings.csv') -NoTypeInformation
$meisdEpoch = 5 * (3*$times.meisd_stl + 3*$times.meisd_pair +
    $times.meisd_hard + $times.meisd_triple + $times.meisd_adapters + $times.meisd_mmoe)
$brighterEpoch = 5 * (2*$times.brighter_stl + $times.brighter_hard +
    $times.brighter_soft + $times.brighter_adapters + $times.brighter_mmoe)
$hours3 = ($meisdEpoch + $brighterEpoch) * 3 / 3600
$hours6 = ($meisdEpoch + $brighterEpoch) * 6 / 3600
$planningHours = 1.5*$hours6 + 4 # 50% marginesu + 4 h na test, raport i I/O
[pscustomobject]@{ ThreeEpochHours=[math]::Round($hours3,1);
    SixEpochHours=[math]::Round($hours6,1);
    PlanningHours=[math]::Round($planningHours,1);
    AvailableWindowHours=48 }
```

Dla okna 48 h **`PlanningHours <= 40`** pozostawia co najmniej
8 h marginesu. Jeśli prognoza przekracza 40 h, wstrzymaj pełny start
i ustal harmonogram lub zakres przed oceną testu. Zapisz tabelę
`timings.csv` i wersję kodu przy wynikach.

## 6. Warunek startu

**Zielone — można rozpocząć pełną macierz tylko gdy:**

- [ ] `main` jest czyste, commit zapisany, GPU/ROCm widoczne, pakiety bez
  konfliktów, wszystkie testy zaliczone.
- [ ] Trzy źródłowe SHA-256 MEISD są dokładne; oba manifesty mają
  1378/118/118 i 2753/115/2765, właściwą rewizję, `start` dla MEISD,
  dev/test bez augmentacji i brak przenikania grup/tekstów.
- [ ] Ustalono limit **192 tokenów**.
- [ ] Wszystkie 11 nowych prób przy 192 tokenach mają
  `status=trained`, checkpoint zgodny z hashem, finite straty,
  diagnostykę i routing MMoE. Wszystkie mieszczą się przy partii 16.
- [ ] MEISD soft sharing z trzema zadaniami przeszedł. Szczyt
  przydzielonego VRAM jest poniżej około 80% pojemności. Przy 80–90%
  wykonaj dodatkową kontrolę stabilności; powyżej 90% wstrzymaj start.
- [ ] Kilkuepokowe próby wyjaśniają kolaps intensywności lub po kontroli
  potwierdzono, że nie wynika z błędu danych/maski/straty. Nie wymagamy
  arbitralnego progu F1: rzetelny negatywny wynik jest dopuszczalny.
- [ ] Na `D:` jest co najmniej 100 GiB, na `C:` co najmniej 10 GiB,
  a `PlanningHours <= 40`. Manifesty zapisują HIP i GPU; zaplanowano
  kontrolę miejsca i archiwizację.

**Żółte — najpierw wyjaśnij:** `low` nadal dominuje po kilku epokach
w jednym modelu, 80–90% VRAM, czasy pilotów są niestabilne lub brak
zapisu HIP/GPU. Słaby F1 sam w sobie nie jest automatycznym `no-go`.

**Czerwone — wstrzymaj macierz:** brak lub błędny hash danych, przenikanie
splitów, CPU zamiast GPU, nieudane testy, OOM, NaN/Inf, niekompletny
checkpoint/diagnostyka, kolaps intensywności w obu dłuższych próbach
bez wyjaśnienia, niewystarczający dysk lub `PlanningHours > 40` bez
zmienionego planu. Jeśli partia 16 nie mieści się w VRAM, ustal nową
partię i akumulację, zapisz zmianę protokołu i ponów pomiary **przed**
wygenerowaniem oficjalnej macierzy.

## 7. Dopiero po zielonym świetle

Generuj macierze na docelowym komputerze: konfiguracje zapisują
bezwzględne ścieżki danych. Nie przenoś plików macierzy z innego
komputera. Nowe katalogi muszą być puste.

```powershell
& .\.venv\Scripts\att.exe make-matrix --dataset data/prepared/meisd-study2-v1 --dataset-name meisd --output outputs/matrix-meisd-study2-2026-10-02
if ($LASTEXITCODE -ne 0) { throw 'Nie utworzono macierzy MEISD' }
& .\.venv\Scripts\att.exe make-matrix --dataset data/prepared/brighter-study2-v1 --dataset-name brighter --output outputs/matrix-brighter-study2-2026-10-02
if ($LASTEXITCODE -ne 0) { throw 'Nie utworzono macierzy BRIGHTER' }
$m = Get-Content outputs/matrix-meisd-study2-2026-10-02/matrix.json -Raw | ConvertFrom-Json
$b = Get-Content outputs/matrix-brighter-study2-2026-10-02/matrix.json -Raw | ConvertFrom-Json
if ($m.runs -ne 50 -or $b.runs -ne 30 -or -not $m.primary_scope -or -not $b.primary_scope) { throw 'Nieprawidłowy zakres macierzy' }
$configs = @(Get-ChildItem outputs/matrix-meisd-study2-2026-10-02/*.json | Where-Object Name -ne 'matrix.json') + @(Get-ChildItem outputs/matrix-brighter-study2-2026-10-02/*.json | Where-Object Name -ne 'matrix.json')
if ($configs.Count -ne 80 -or @($configs | Where-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).max_length -ne 192 }).Count) { throw 'Macierz nie używa 192 tokenów' }
git rev-parse HEAD
```

Po zapisaniu pomiarów i decyzji uruchom według [`RUN_HANDOFF.md`](RUN_HANDOFF.md).
Polecenia poniżej są podane dla kompletności — **nie wykonuj ich podczas
kontroli przedstartowej**:

```powershell
& .\.venv\Scripts\att.exe run-matrix --matrix outputs/matrix-meisd-study2-2026-10-02/matrix.json
& .\.venv\Scripts\att.exe run-matrix --matrix outputs/matrix-brighter-study2-2026-10-02/matrix.json
```

`run-matrix` sprawdza ukończone checkpointy i wznawia zgodny, przerwany
run z ostatniej zapisanej epoki. Częściowo wykonana epoka jest liczona
ponownie. `history.jsonl` i `diagnostics.jsonl` zapisują wyniki w trakcie,
a przejściowy `last_epoch.pt` zawiera model, optymalizator, scheduler
i stan losowości; po ukończeniu runu jest usuwany. Nie kasuj katalogu
po błędzie. Terminal pokazuje postęp każdej macierzy osobno (50 MEISD
i 30 BRIGHTER) oraz orientacyjny pozostały czas po pierwszym ukończonym
runie. Aktualny stan jest w `progress/status.json` w katalogu macierzy;
prognoza zmienia się, gdy zaczynają się wolniejsze architektury.
Ocenę testową i raport wykonaj dopiero po przeglądzie całej
zaplanowanej macierzy:

```powershell
& .\.venv\Scripts\att.exe evaluate-matrix --matrix outputs/matrix-meisd-study2-2026-10-02/matrix.json --split test
if ($LASTEXITCODE -ne 0) { throw 'Niekompletna ocena MEISD' }
& .\.venv\Scripts\att.exe evaluate-matrix --matrix outputs/matrix-brighter-study2-2026-10-02/matrix.json --split test
if ($LASTEXITCODE -ne 0) { throw 'Niekompletna ocena BRIGHTER' }
& .\.venv\Scripts\att.exe report --matrices outputs/matrix-meisd-study2-2026-10-02/matrix.json outputs/matrix-brighter-study2-2026-10-02/matrix.json --expected-runs 80 --output outputs/report-study2 --split test
if ($LASTEXITCODE -ne 0) { throw 'Niekompletny raport' }
```

Raport wymaga dokładnie wskazanych macierzy, wszystkich wyników testu,
zgodnych konfiguracji i hashy. Nie skanuje przypadkowych pilotów.
Archiwizuj `outputs/` osobno od Git; pełnych checkpointów nie commituj.
Dodatkowe backbone'y, architektury historyczne i ablacje wymagają
oddzielnego planu czasu i miejsca.
