# CLAUDE.md - python_migration work log

Project: move the Deep Forecasting (DATA 5630/6630) PyCaret notebooks off official `pycaret` 3.3.2, which no longer imports on Python >= 3.12 (Colab is 3.13 now). Owner: Pedram Jahangiry. Started 2026-09-16.
Destination: this folder is meant to be pushed to the course GitHub repo (`PJalgotrader/Deep_forecasting-USU`) as `python_migration/` with three student options: colab, uv, conda.

Read `README.md` for the student-facing story; this file is for whoever works on the migration next (human or Claude).

**Where things live:** the repo copy of this folder holds the three options, the tutorial and the Colab test notebooks. The `evidence/` folder (smoke logs, uv resolutions, Colab leaderboard CSV), `video_script.md`, and `backup_notebooks_2026-09-16/` referenced below stay on Pedram's Drive at `DF Lectures/python_migration/` and are deliberately not committed.

In the repo the uv option replaced the old `environments/pycaret/` project (`git mv`, 2026-09-16); the main course env at the repo root (Python 3.11, no pycaret) is unchanged.

## Decisions (do not re-litigate)

| Decision | Why | Evidence |
|---|---|---|
| Use **pycaret-core 3.5.0** (sktime-org community fork, MIT) | Same `import pycaret`, same `TSForecastingExperiment` API; the only notebook change is the install line | `evidence/smoke_results.log`, `colab_tests/B_*.ipynb` |
| Pin **`statsmodels<0.15`** | statsmodels 0.15.0 (2026-08-27) renamed `ETSResults.simulate(random_state=)` to `rng=`; sktime 1.1.0 still passes `random_state`, so `create_model('ets')` fails. Fix merged in sktime PR 10972 on 2026-09-13, not released yet | `evidence/ets_only.py` run in `smoke_results.log` |
| **Not pycaret 4.0** (4.0.0a8, May 2026) | Rewrite: `TSForecastingExperiment` -> `TimeSeriesExperiment`, functional API, `plot_model`, `check_stats`, `evaluate_model` removed; also breaks Colab's ipython pin | `colab_tests/C_*.ipynb`, `evidence/smoke_v4.py` |
| **Python 3.13** for uv and conda | Matches Colab (3.13.15 on 2026-09-16) and puts the fork on its numpy 2 branch. On Python < 3.13 the fork pins numpy<2, pandas<2.2, scikit-learn<1.5, which clashes with the Nixtla notebooks | `evidence/out_pycaret-core__3_5_0_3.1{2,3}.txt` |
| Keep pycaret in its **own environment** | Because of the pins above; the Nixtla/statsforecast notebooks stay in their own env | question bank m2_p1 Q17 already says this |
| **shap added to the uv env** (2026-09-16, step 7) | `Platforms and tools/PyCaret/PyCaret-RegressionDemo.ipynb` calls `interpret_model`, which needs shap; the old `environments/pycaret` had it through `pycaret[full]`. Not added to the Colab line or conda file (time-series notebooks never need it) | regression demo re-run on the fork |
| **lightgbm, xgboost, catboost listed explicitly** in all three options | So `compare_models()` shows the same rows everywhere; lightgbm is a hard dep of pycaret-core but Pedram wants it guaranteed | `evidence/step2_*_models.log` |

## Status

- [x] Step 1 - live Colab test of 3.3.2 / pycaret-core / 4.0 (`colab_tests/`, `evidence/colab_results.md`); B re-run 2026-09-16 15:45 with the final install line and `compare_models()` over all 28 models (`evidence/colab_leaderboard_all_models.csv`)
- [x] Step 2 - `colab/`, `uv/` (pyproject + uv.lock), `conda/` (environment.yml) built and smoke-tested locally (`evidence/step2_summary.md`); VS Code kernel instructions added to both READMEs (Pedram: students must be able to run in VS Code)
- [x] Step 3 - seven notebooks updated and re-executed on the uv lock, `.pkl` files regenerated (`evidence/step3_summary.md`; originals in `backup_notebooks_2026-09-16/`). Two content fixes: All-in-one `tune_model(xgboost)` -> `tune_model(rf)`; SARIMAX cell 35 ValueError is pre-existing and kept
- [x] Step 4 - `tutorial.html` (self-contained, light/dark, phone-safe; Google Fonts only external dependency) and `video_script.md` (12-min scene plan with on-screen actions, B-roll list, thumbnail ideas). Update both if pins or versions change. Also `conda_to_uv_student_cheatsheet.html`: Pedram's beginner uv cheat sheet, expanded 2026-09-16 (the three files, updating, troubleshooting, course-project section); original kept in `backup_notebooks_2026-09-16/`.
- [x] Step 5 - question bank m2_p1 Q17 rewritten around `pycaret-core` + the statsmodels pin (Drive only: `DATA5630_question_bank/m2_bank/m2_p1_Qbank.md`, PDF regenerated; `review/build_question_bank_pdfs.py` font path fixed to fall back to matplotlib's DejaVu fonts)
- [x] Step 6 - old `DF_environment.yml` moved to `backup_notebooks_2026-09-16/` (Drive). Module 2 deck (`Module 2-DF environment-original.pptx`, Drive) edited: PyCaret install bullet + install line updated, new slide 8 "Three ways to run the course notebooks" inserted after the Colab slide; PDF exported with PowerPoint and copied to the repo as `Lectures and codes/Module 2- Setting up DF environment/Module 2-DF environment.pdf`
- [x] Step 7 - `python_migration/` added to the course repo on branch `pycaret-core-2026` (`environments/pycaret` -> `python_migration/uv` via git mv; five notebooks replaced with fixed Colab badges; README, `.gitignore`, `Platforms and tools/PyCaret` and `Platforms and tools/uv` updated). Merge + push pending Pedram's review.

## Step 3 scope - notebooks that run pycaret (done 2026-09-16; kept for reference)

| Notebook | Install cells | Note |
|---|---|---|
| `All in one.ipynb` | cells 4-6 | `version()` output shows 3.0.0 |
| `Module 3- Exponential Smoothing/Module3-exponential_smoothing_ETS.ipynb` | cells 4-6 | heaviest user (ETS - needs the statsmodels pin); writes `best_smoothing_model.pkl` |
| `Module 4- ARIMA/Module4-ARIMA.ipynb` | cells 4-6 | writes `archive/best_arima_model.pkl` |
| `Module 4- ARIMA/Module4-SARIMAX.ipynb` | cells 4-6 | `version()` output shows 3.2.0 |
| `Module 5- Machine Learning Forecasting/Module5-ML_timesereis.ipynb` | none - add one | uses sktime `WindowSummarizer` via `fe_target_rr` (verified on sktime 1.1.0) |
| `Misellanuous/stock_market/Predicting_stock_price_PyCaret.ipynb` | cells 4-6 + cell 23 | cell 23 is a stale `# !pip install packaging==21.3` - drop it |
| `Misellanuous/stock_market/Predictiong_stock_returns_PyCaret.ipynb` | cells 4-6 | |

Cell layout in the six with an install section: [4] markdown link to pycaret gitbook, [5] commented `!pip install pycaret...`, [6] `from pycaret.utils import version; version()` with the "RESTART RUNTIME" comment. Replacement install line: `!pip install -q pycaret-core "statsmodels<0.15" lightgbm xgboost catboost`. The Colab "restart runtime" advice is obsolete on Python 3.13 (nothing preinstalled gets downgraded).

`.pkl` files saved by 3.3.2 / sktime 0.26 should be regenerated under sktime 1.1.0 rather than trusted to load.

## Conventions and gotchas

- `Platforms and tools/PyCaret/PyCaret-ClassificationDemo.ipynb` was last run on pycaret 3.0.0.rc4; its `group_features` used the PyCaret 2 list-of-lists form. Converted to the PyCaret 3 dict form `{name: [cols]}` on 2026-09-16 (fork-independent drift).
- **Never build environments inside this Drive folder.** `I:\My Drive` is Google Drive for desktop; a `.venv` here would sync thousands of files. Locally use `UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\Temp\pcsmoke\uv_env` (or any short path). Students on GitHub are fine with the default `.venv` (gitignored).
- **Windows MAX_PATH:** uv installs fail with "Failed to persist temporary file ... os error 3" when the venv path is long. Use a short temp dir for throwaway envs.
- **Run smoke tests from a temp cwd**, not from here: pycaret writes `logs.log`, `catboost_info/`, and `.pkl` files into the working directory (all gitignored anyway).
- Smoke suite: `evidence/smoke_ts_v2.py` (29 checks mirroring the notebooks' API surface) and `evidence/check_models.py` (GBM forecasters). Run them against any new env before changing pins.
- **Install before importing pycaret, and use a fresh Colab runtime.** pycaret checks soft dependencies (catboost, xgboost, prophet...) once at first import and caches the result; a `pip install` later in the same kernel is invisible even after `importlib.invalidate_caches()` (reproduced 2026-09-16). Symptom: `ValueError: Estimator catboost_cds_dt Not Available`. Fix: Runtime > Disconnect and delete runtime, then Run all. A tell-tale sign of a stale runtime is the "before install" cell already listing pycaret-core.
- `colab_tests/*.ipynb` are regenerated by `build_colab_tests.py`; regenerating wipes the executed outputs, so do not run it casually.
- Local baseline for before/after comparisons: conda env `deep_forecasting` (Python 3.11.9, pycaret 3.3.2, sktime 0.26.0, statsmodels 0.14.2). Local product env: conda `df_pycaret`.
- Verified versions on Colab (2026-09-16): Python 3.13.15, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.3, statsmodels 0.15.0. The public runtime FAQ table lags reality; always check `sys.version` in a fresh runtime.

## Things to revisit later

- When sktime releases a version > 1.1.0 containing PR 10972, test dropping the `statsmodels<0.15` pin (`uv lock --upgrade-package sktime --upgrade-package statsmodels`, rerun the smoke suite), then update all three options and the notebooks' comments.
- pycaret-core is five weeks into "resumed community maintenance" (v3.5.0, 2026-08-09). Check https://github.com/sktime/pycaret/releases each semester.
- Longer term: Modules 3, 4, 5 already have statsforecast / mlforecast sibling notebooks; All-in-one, SARIMAX, and the two stock-market notebooks do not. If pycaret-core stalls, that is the exit path.
