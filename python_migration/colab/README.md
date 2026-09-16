# Option 1 - Google Colab

Colab's default runtime is Python 3.13 (checked 2026-09-16). The official `pycaret` 3.3.2 package refuses to import on Python >= 3.12, so the course now installs the community fork `pycaret-core`, which keeps the identical API.

Replace the old install cell (`!pip install pycaret` / `!pip install pycaret[full]`) with:

```python
!pip install -q pycaret-core "statsmodels<0.15" lightgbm xgboost catboost
```

Then verify:

```python
from pycaret.utils import version
version()          # -> '3.5.0'
```

Notes
- No runtime restart is needed: on Python 3.13 the fork keeps Colab's numpy 2 / pandas 2.2 / scikit-learn 1.6.
- `statsmodels<0.15` is required until sktime ships the fix for statsmodels 0.15 (sktime PR 10972, merged 2026-09-13). Without it, `create_model('ets')` fails.
- Do **not** use `pip install --pre pycaret` (PyCaret 4.0 alpha): it renames `TSForecastingExperiment` and removes `plot_model` and `check_stats`.
