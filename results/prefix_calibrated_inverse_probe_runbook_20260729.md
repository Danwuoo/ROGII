# Prefix-calibrated inverse probe runbook

All commands below are PowerShell commands and were checked with `--help`,
focused tests, smoke, or the completed disk artifacts. Run them from the
repository root.

## Environment and data check

```powershell
Set-Location 'C:\Users\wurre\Desktop\ROGII'
$DataRoot = 'data\rogii-wellbore-geology-prediction'
$RunDir = '.runs\prefix-calibrated-inverse-probe-20260729'
$CacheDir = '.runs\cache\prefix-inverse'

rogii check --data-root $DataRoot
python scripts\run_prefix_calibrated_inverse_probe.py --help
python scripts\run_prefix_calibrated_inverse_probe.py `
  --stage environment --profile smoke --device auto `
  --data-root $DataRoot --run-dir $RunDir
```

## CPU smoke

The existing smoke artifact is resumable:

```powershell
python scripts\run_prefix_calibrated_inverse_probe.py `
  --stage observability --profile smoke --device cpu --seed 20260729 `
  --data-root $DataRoot `
  --run-dir "$RunDir\smoke" `
  --cache-dir '.runs\cache\prefix-inverse-smoke' `
  --num-workers 1 --grid-chunk-rows 256 --memory-limit-mb 1024 `
  --resume
```

For a new smoke directory, omit `--resume`.

## Stage 0 observability

The cold command used for the completed 773-well artifact was:

```powershell
python scripts\run_prefix_calibrated_inverse_probe.py `
  --stage observability --profile quick --device cpu --seed 20260729 `
  --data-root $DataRoot --run-dir $RunDir --cache-dir $CacheDir `
  --num-workers 1 --grid-chunk-rows 2048 --memory-limit-mb 4096
```

Do not rerun it with altered parameters. The locked result failed.

## Exact resume verification

```powershell
python scripts\run_prefix_calibrated_inverse_probe.py `
  --stage observability --profile quick --device cpu --seed 20260729 `
  --data-root $DataRoot --run-dir $RunDir --cache-dir $CacheDir `
  --num-workers 1 --grid-chunk-rows 2048 --memory-limit-mb 4096 `
  --resume
```

## Independent disk audit

```powershell
python scripts\audit_prefix_calibrated_inverse_probe.py `
  --run-dir $RunDir
```

## Visible-test offline smoke

```powershell
python scripts\run_prefix_calibrated_inverse_probe.py `
  --stage visible-test-smoke --profile smoke --device cpu --seed 20260729 `
  --data-root $DataRoot --run-dir $RunDir `
  --cache-dir '.runs\cache\prefix-inverse-visible' `
  --num-workers 1 --grid-chunk-rows 256 --memory-limit-mb 1024
```

This validates dynamic test enumeration and sample ID coverage. It does not
write a submission.

## Quick prediction fail-closed check

Stage 0 failed, so this verified command exits nonzero before prediction:

```powershell
python scripts\run_prefix_calibrated_inverse_probe.py `
  --stage prediction --profile quick --device cpu `
  --data-root $DataRoot --run-dir $RunDir
```

Expected message:

```text
Stage 0 did not pass; full inverse prediction is fail-closed
```

## Tests

```powershell
pytest -q `
  tests\test_prefix_calibration.py `
  tests\test_inverse_observability.py `
  tests\test_inverse_fold_safety.py `
  tests\test_inverse_probe_end_to_end.py

.\.runs\ttm-r2-transfer-probe-20260729\.venv\Scripts\python.exe `
  -m pytest -q
```

## Result summary

```powershell
Get-Content -Raw "$RunDir\stage0_gate.json"
Get-Content -Raw "$RunDir\stage0_diagnostics.json"
Get-Content -Raw "$RunDir\summary.json"
Get-Content -Raw "$RunDir\independent_audit.json"
Get-Content -Raw 'results\prefix_calibrated_inverse_probe_20260729.md'
```

## Full commands

No full, full-resume, or full-audit command is provided. The locked Stage 0
gate failed, and invoking full would violate the protocol. No submission
creation or upload command exists in this runbook.
