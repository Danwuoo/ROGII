# ROGII Kaggle notebooks

Only two thin offline launchers are active:

- `flat.ipynb`
- `p3.ipynb`

Each notebook installs the packaged `rogii` wheel, runs complete five-fold CV,
then calls `rogii submit`. Model code lives exclusively in `src/rogii/`.

Attach these fixed inputs with internet disabled:

- Runtime dataset: `/kaggle/input/datasets/danielwuu/upload/rogii_kaggle_dataset`
- Competition data: `/kaggle/input/competitions/rogii-wellbore-geology-prediction`

The runtime dataset must contain:

```text
wheels/rogii-*.whl
configs/folds/main_5fold.csv
```

The generated file is `/kaggle/working/submission.csv`.
