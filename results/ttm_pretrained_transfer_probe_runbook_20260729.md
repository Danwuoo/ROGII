# Granite TTM R2 pretrained-transfer probe runbook

This runbook is for PowerShell in
`C:\Users\wurre\Desktop\ROGII`. Commands below were checked with `--help`,
smoke, quick, resume, or independent audit on 2026-07-29.

The actual quick gate failed. The full commands are retained for a future
independently passed/frozen quick protocol, but the current CLI intentionally
refuses them while the sibling `quick/quick_gate.json` says `passed=false`.

## Paths

```powershell
$Root = 'C:\Users\wurre\Desktop\ROGII'
Set-Location $Root
$Base = Join-Path $Root '.runs\ttm-r2-transfer-probe-20260729'
$Venv = Join-Path $Base '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$CacheDir = Join-Path $Base 'cache'
$HfCache = Join-Path $Base 'hf-cache'
$WeightDir = Join-Path $Base 'weights\ibm-granite--granite-timeseries-ttm-r2--d6a79570'
$DataRoot = Join-Path $Root 'data\rogii-wellbore-geology-prediction'
$Folds = Join-Path $Root 'configs\folds\main_5fold.csv'
$Locked = Join-Path $Root 'configs\external\ttm_r2_transfer_probe_20260729.json'
$IncumbentRoot = Join-Path $Root '.runs\heel-registered-anchor-full-20260728'
$IncumbentOof = Join-Path $IncumbentRoot 'oof_predictions.csv.gz'
$QuickSource = Join-Path $Root '.runs\sparse-anchor-student-quick\run_config.json'
```

## 1. Create/activate the isolated environment

The tested environment uses `--system-site-packages` so the project keeps its
scikit-learn 1.8.0 incumbent runtime. PyPI `granite-tsfm==0.3.6` declares
`scikit-learn<1.8`; its local 1.7.2 is removed after installation. The TTM
backbone path does not call scikit-learn, and this exact arrangement passed all
smokes, exact incumbent comparison, quick, and tests.

```powershell
if (-not (Test-Path $Venv)) {
    py -3.12 -m venv --system-site-packages $Venv
}
& $Py -m pip install --upgrade pip
& $Py -m pip install 'granite-tsfm==0.3.6' 'transformers==4.57.6' 'huggingface-hub==0.36.2' 'safetensors==0.8.0' 'torch==2.10.0'
& $Py -m pip uninstall -y scikit-learn
& $Py -m pip install --no-deps 'scikit-learn==1.8.0'
& $Py -m pip install -e . --no-deps
& $Py -c "import sklearn, torch, transformers, importlib.metadata as m; assert sklearn.__version__ == '1.8.0'; assert torch.__version__.startswith('2.10.0'); assert transformers.__version__ == '4.57.6'; assert m.version('granite-tsfm') == '0.3.6'; print('TTM environment OK')"
```

Do not install this optional stack into the production Python environment.
The audited package manifest is
`configs/external/ttm_r2_transfer_environment_20260729.json`.

## 2. Verify the official repository pin

The existing audit clone is under ignored `.runs/`.

```powershell
$Vendor = Join-Path $Base 'vendor\granite-tsfm'
$RepoHead = (git -C $Vendor rev-parse HEAD).Trim()
if ($RepoHead -ne '31c7a6bd640c41e53777c0afe3a13d24e00f7fe7') {
    throw "Granite repository revision mismatch: $RepoHead"
}
```

The Windows sparse audit clone may report one excluded notebook PNG as modified
because its path exceeds normal Windows path handling. Model/package source,
HEAD identity, and weights are unaffected; the weight identity is independently
verified below.

## 3. Download only the fixed checkpoint revision

The downloader allowlists `README.md`, `config.json`,
`generation_config.json`, and `model.safetensors`, requires both cache and
weights below project `.runs/`, and verifies the checkpoint immediately.

```powershell
& $Py scripts\download_ttm_r2_weights.py `
    --cache-dir $HfCache `
    --weight-dir $WeightDir
```

No pretraining dataset is downloaded.

## 4. Verify the weight SHA-256

```powershell
$ExpectedWeightHash = 'a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710'
$ActualWeightHash = (Get-FileHash -Algorithm SHA256 (Join-Path $WeightDir 'model.safetensors')).Hash.ToLowerInvariant()
if ($ActualWeightHash -ne $ExpectedWeightHash) {
    throw "TTM weight hash mismatch: $ActualWeightHash"
}
$ActualWeightHash
```

Expected size is 3,240,592 bytes. Any mismatch fails closed.

## 5. Online smoke

The conditional `--resume` makes this command safe both for the existing
workspace and for a new empty smoke directory.

```powershell
$SmokeOnline = Join-Path $Base 'smoke-online'
$SmokeOnlineResume = @()
if (Test-Path (Join-Path $SmokeOnline 'run_config.json')) {
    $SmokeOnlineResume = @('--resume')
}
& $Py scripts\run_ttm_transfer_probe.py `
    --profile smoke `
    --run-dir $SmokeOnline `
    --cache-dir $CacheDir `
    --weight-dir $WeightDir `
    --data-root $DataRoot `
    --device cpu `
    --num-workers 0 `
    --batch-size 2 `
    --embedding-batch-size 2 `
    @SmokeOnlineResume
```

## 6. Forced-offline smoke

This sets the library offline path and also replaces DNS/socket entry points
with fail-closed guards during model loading/forward execution.

```powershell
$SmokeOffline = Join-Path $Base 'smoke-offline'
$SmokeOfflineResume = @()
if (Test-Path (Join-Path $SmokeOffline 'run_config.json')) {
    $SmokeOfflineResume = @('--resume')
}
& $Py scripts\run_ttm_transfer_probe.py `
    --profile smoke `
    --run-dir $SmokeOffline `
    --cache-dir $CacheDir `
    --weight-dir $WeightDir `
    --data-root $DataRoot `
    --device cpu `
    --num-workers 0 `
    --batch-size 2 `
    --embedding-batch-size 2 `
    --offline `
    @SmokeOfflineResume

& $Py scripts\audit_ttm_transfer_probe.py `
    --run-dir $SmokeOffline `
    --output-path (Join-Path $SmokeOffline 'independent_audit.json')
```

## 7. Canonical 100-well quick

The current canonical quick is complete. This command resumes the existing
run; remove `--resume` only when using an empty, new run directory under a new
predeclared protocol.

```powershell
$Quick = Join-Path $Base 'quick'
& $Py scripts\run_ttm_transfer_probe.py `
    --profile quick `
    --run-dir $Quick `
    --cache-dir $CacheDir `
    --weight-dir $WeightDir `
    --data-root $DataRoot `
    --fold-config $Folds `
    --locked-config $Locked `
    --incumbent-root $IncumbentRoot `
    --incumbent-oof $IncumbentOof `
    --quick-source $QuickSource `
    --seed 20260729 `
    --seed 20260730 `
    --seed 20260731 `
    --device cpu `
    --num-workers 4 `
    --batch-size 16 `
    --embedding-batch-size 32 `
    --bootstrap-samples 10000 `
    --offline `
    --resume
```

For a genuinely fresh quick, use the same command with a new empty run
directory and omit `--resume`. Do not delete or overwrite the audited current
quick.

## 8. Quick resume after interruption

Use the exact same command/config/seeds as section 7 and add `--resume`.
Completed checkpoints are loaded safely; incomplete epoch checkpoints continue
with optimizer state and deterministic epoch/batch order. A config, source
hash, fold, seed, data manifest, or provenance mismatch is rejected.

The audited cold CPU quick took 18m 37s. Exact resume took 5m 44s and reproduced
the identical OOF SHA-256.

## 9. Independent quick audit

```powershell
& $Py scripts\audit_ttm_transfer_probe.py `
    --run-dir $Quick `
    --output-path (Join-Path $Quick 'independent_audit.json')
```

This re-reads the 154 MB OOF artifact and all 90 checkpoint/prediction-block
hashes, rechecks folds/keys/finite values/incumbent exactness, recomputes
metrics/gate, and writes per-seed/per-fold metrics.

## 10. View the quick gate

```powershell
$Gate = Get-Content (Join-Path $Quick 'quick_gate.json') -Raw | ConvertFrom-Json
$Gate | ConvertTo-Json -Depth 12
```

Current expected result:

```text
passed   = false
decision = stop_ttm_no_full_no_adapter_no_retune
```

On failure, stop. Do not run full, adapter, scratch-full, nested ensemble, or
retune/retry on this cohort.

## 11. Full five-seed training — conditional command, currently prohibited

The following is the complete command for a future passed sibling quick. In the
current workspace the CLI exits before reading the 773-well data because the
locked quick gate failed.

```powershell
$Full = Join-Path $Base 'full'
$Gate = Get-Content (Join-Path $Quick 'quick_gate.json') -Raw | ConvertFrom-Json
if ($Gate.passed -ne $true) {
    throw 'Do not run full: the locked TTM quick transfer gate failed.'
}
& $Py scripts\run_ttm_transfer_probe.py `
    --profile full `
    --run-dir $Full `
    --cache-dir $CacheDir `
    --weight-dir $WeightDir `
    --data-root $DataRoot `
    --fold-config $Folds `
    --locked-config $Locked `
    --incumbent-root $IncumbentRoot `
    --incumbent-oof $IncumbentOof `
    --quick-source $QuickSource `
    --seed 20260711 `
    --seed 20260712 `
    --seed 20260713 `
    --seed 20260714 `
    --seed 20260715 `
    --device auto `
    --num-workers 4 `
    --batch-size 16 `
    --embedding-batch-size 32 `
    --bootstrap-samples 10000 `
    --offline
```

`--profile full` is mandatory and has no default. Full defaults to only the
matched random-frozen and pretrained-frozen families. Adapter and scratch-full
are not included.

## 12. Full resume — conditional

```powershell
$Gate = Get-Content (Join-Path $Quick 'quick_gate.json') -Raw | ConvertFrom-Json
if ($Gate.passed -ne $true) {
    throw 'Do not resume full: the locked TTM quick transfer gate failed.'
}
& $Py scripts\run_ttm_transfer_probe.py `
    --profile full `
    --run-dir $Full `
    --cache-dir $CacheDir `
    --weight-dir $WeightDir `
    --data-root $DataRoot `
    --fold-config $Folds `
    --locked-config $Locked `
    --incumbent-root $IncumbentRoot `
    --incumbent-oof $IncumbentOof `
    --quick-source $QuickSource `
    --seed 20260711 `
    --seed 20260712 `
    --seed 20260713 `
    --seed 20260714 `
    --seed 20260715 `
    --device auto `
    --num-workers 4 `
    --batch-size 16 `
    --embedding-batch-size 32 `
    --bootstrap-samples 10000 `
    --offline `
    --resume
```

## 13. Independent full audit — conditional

```powershell
$Gate = Get-Content (Join-Path $Quick 'quick_gate.json') -Raw | ConvertFrom-Json
if ($Gate.passed -ne $true) {
    throw 'Do not audit a nonexistent full: the locked TTM quick gate failed.'
}
& $Py scripts\audit_ttm_transfer_probe.py `
    --run-dir $Full `
    --output-path (Join-Path $Full 'independent_audit.json')
```

Nested ensemble and 100,000-draw shadow-private simulation remain downstream
of a credible standalone full result. They are not authorized by the failed
quick and were not run.

## 14. Visible-test forced-offline smoke

This is label-free and writes only JSON audit metadata.

```powershell
& $Py scripts\smoke_ttm_visible_test.py `
    --run-dir $Quick `
    --output-path (Join-Path $Quick 'visible_test_offline_smoke.json') `
    --device cpu `
    --offline `
    --num-workers 4 `
    --batch-size 16 `
    --embedding-batch-size 32
```

It dynamically enumerates test wells and joins/sorts by
`sample_submission.csv` IDs. It does not create a submission CSV.

## 15. Focused and complete tests

```powershell
& $Py -m pytest -q `
    tests\test_ttm_embeddings.py `
    tests\test_ttm_fold_safety.py `
    tests\test_ttm_transfer_probe.py `
    tests\test_ttm_offline_smoke.py

& $Py -m pytest -q
rogii check --data-root $DataRoot
```

## 16. Produce a compact handoff for the next chat

The checked-in quick report is already a compact, human-readable handoff:

```powershell
Get-Content results\ttm_pretrained_transfer_probe_quick_20260729.md -Raw
```

Return these small files:

```powershell
$Handoff = @(
    'results\ttm_pretrained_transfer_probe_quick_20260729.md',
    '.runs\ttm-r2-transfer-probe-20260729\quick\quick_gate.json',
    '.runs\ttm-r2-transfer-probe-20260729\quick\independent_audit.json',
    '.runs\ttm-r2-transfer-probe-20260729\quick\visible_test_offline_smoke.json',
    '.runs\ttm-r2-transfer-probe-20260729\quick\metrics\summary.json',
    '.runs\ttm-r2-transfer-probe-20260729\quick\metrics\per_fold.csv',
    '.runs\ttm-r2-transfer-probe-20260729\quick\metrics\per_seed_fold.csv'
)
$Handoff | ForEach-Object {
    Get-Item (Join-Path $Root $_) | Select-Object FullName, Length
}
```

Do not send the 154 MB OOF, checkpoint tree, embedding cache, or prediction
blocks unless specifically requested.

## Runtime and memory

- Audited CPU-only quick cold run: 18m 37s; resume: 5m 44s.
- Audited quick peak host working set: approximately 2.36 GB.
- CPU fallback: verified.
- CUDA option: implemented (`auto|cpu|cuda`) but not executed on this host
  because the installed PyTorch build exposed no CUDA device.
- Estimated CUDA memory for the fixed batch sizes: under 2 GB; reserve 3 GB.
  This is an engineering estimate, not a measured value on this host.
- Conditional full estimate: roughly 1–3 hours CPU or 45–120 minutes on a
  modest CUDA GPU, plus 10–25 minutes for disk audit; 6–10 GB host RAM/disk
  working headroom is prudent. These are extrapolations and full was not run.

## Artifact locations

- Weights:
  `.runs/ttm-r2-transfer-probe-20260729/weights/ibm-granite--granite-timeseries-ttm-r2--d6a79570/`
- Hugging Face download cache:
  `.runs/ttm-r2-transfer-probe-20260729/hf-cache/`
- Prepared sequence/incumbent cache:
  `.runs/ttm-r2-transfer-probe-20260729/cache/`
- Quick checkpoints:
  `.runs/ttm-r2-transfer-probe-20260729/quick/checkpoints/`
- Embedding cache:
  `.runs/ttm-r2-transfer-probe-20260729/quick/embedding_cache/`
- OOF and metrics:
  `.runs/ttm-r2-transfer-probe-20260729/quick/`

All large artifacts remain ignored under `.runs/`. `data/` stayed unchanged.
No Kaggle submission was created or uploaded.
