![DF cover](https://user-images.githubusercontent.com/19335954/210499919-b5000dda-b46c-42b9-b274-fe06116c8260.png)

# Deep Forecasting - Fall 2026
### Advanced Time Series Analysis and Forecasting with Deep Learning
**Utah State University | Huntsman School of Business**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/PJalgotrader/Deep_forecasting-USU)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

> [!IMPORTANT]
> **How to run the course:** Modules 2–5 run locally in **one** environment (`uv sync` at the repo root, or Conda) or on Google Colab. Modules 6–7 run on **Google Colab** (they need a GPU). Module 8 is optional and runs on Colab or in its own small local environment. If you are unsure, begin with Google Colab.
>
> **PyCaret changed in 2026.** The official `pycaret` package no longer runs on Colab's Python. The course now uses the community fork `pycaret-core`; the notebooks already carry the right install cell. Read [`python_migration/`](python_migration/) and its [tutorial](python_migration/tutorial.html) if you want the details.

---

## 📚 Course Overview

This comprehensive course introduces students to state-of-the-art time series forecasting techniques, progressing from classical statistical methods to advanced deep learning architectures. Students will gain hands-on experience with real-world forecasting problems using industry-standard tools and frameworks.

### 🎯 Learning Objectives

Upon completion of this course, students will be able to:
- Master fundamental time series concepts and decomposition techniques
- Implement classical forecasting methods (ETS and SARIMAX)
- Apply machine learning algorithms to time series problems
- Design and train deep neural networks for sequence modeling
- Deploy production-ready forecasting models at scale
- Evaluate and compare model performance using appropriate metrics

---

## 📋 Prerequisites

- **Programming**: Basic Python proficiency (variables, loops, functions)
- **Mathematics**: College-level statistics and linear algebra
- **Software**: Google account for Colab access (no local installation required; an optional one-command local setup is described below)

For students needing a refresher, we provide a comprehensive [Python Crash Course](Lectures%20and%20codes/Module%202-%20Setting%20up%20DF%20environment/Python_Crash_course_2020/) covering:
- Python basics, NumPy, Pandas
- Data visualization (Matplotlib, Seaborn)
- Time series data manipulation

---

## 🗂️ Course Modules

### Module 1: Demystifying Time Series Data and Modeling
- Time series components and patterns
- Stationarity and transformations
- Autocorrelation and partial autocorrelation

### Module 2: Setting up Deep Forecasting Environment
- Python environment configuration
- Essential libraries and tools
- Google Colab setup and best practices

### Module 3: Exponential Smoothing Methods
- Simple, Holt's, and Holt-Winters methods
- ETS (Error, Trend, Seasonal) models
- Model selection and validation

### Module 4: SARIMAX Models
- AR, MA, and ARMA processes
- ARIMA, SARIMA, and SARIMAX
- Exogenous regressors and Box-Jenkins methodology

### Module 5: Machine Learning for Time Series
- Feature engineering for time series
- Tree-based methods (Random Forest, XGBoost, LightGBM)
- Cross-validation strategies

### Module 6: Deep Neural Networks
- Feedforward networks for time series
- Backpropagation and optimization
- TensorFlow/Keras implementation

### Module 7: Deep Sequence Modeling
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM) networks
- Bidirectional and stacked architectures

### Optional Module 8: Prophet and NeuralProphet
- Forecasting at scale
- Handling seasonality and holidays
- Uncertainty quantification

---

## 🛠️ Tools and Platforms

### Primary Frameworks
- **[PyCaret](Platforms%20and%20tools/PyCaret/)**: AutoML for time series, installed as [`pycaret-core`](python_migration/) since 2026
- **[Nixtla](https://nixtlaverse.nixtla.io/)**: Statistical, machine-learning, and neural forecasting libraries
- **[TensorFlow/Keras](https://www.tensorflow.org/)**: Deep learning
- **[Prophet/NeuralProphet](https://facebook.github.io/prophet/)**: Scalable forecasting
- **[Streamlit](Platforms%20and%20tools/streamlit/)**: Interactive dashboards

### Development Environment
- **[Google Colab](Platforms%20and%20tools/Google%20Colab/)**: Cloud-based Jupyter notebooks
- **GitHub**: Version control and collaboration
- **Requirements**: Modern web browser, stable internet connection

---

## 📊 Datasets

The course includes various real-world datasets:
- **Airline Passengers**: Classic time series dataset
- **Retail Sales**: Rossmann store sales data
- **Economic Indicators**: US GDP and macroeconomic data
- **Stock Market**: Financial time series examples
- **Custom Projects**: Students can bring their own data

All datasets are available in the [`data/`](data/) directory.

---

## 💻 Getting Started

### Which setup for which module?

| Modules | Where to run | Setup |
|---|---|---|
| **2 – 5** (time-series basics, ETS, ARIMA, ML forecasting, PyCaret) | Google Colab **or** your computer | Colab: nothing to install. Local: the **one** course environment below (`uv` or Conda) |
| **6 – 7** (deep neural networks, RNN/LSTM) | **Google Colab** | Nothing to install. Use a GPU runtime (Runtime → Change runtime type → GPU) |
| **8** (Prophet / NeuralProphet, optional) | Google Colab, or a small local environment of its own | See [Module 8](#module-8-optional-prophet-and-neuralprophet) below |
| Nixtla material (final project) | Its own environment | [`miscellaneous/nixtla/requirements.txt`](Lectures%20and%20codes/miscellaneous/nixtla/requirements.txt) |

### Option 1: Google Colab — easiest

1. Open a notebook on GitHub.
2. Click its **Open in Colab** badge.
3. Sign in with your Google account.
4. Run cells from top to bottom with `Shift+Enter`.

Package-install cells inside notebooks are intended for Colab and run only there.

For the PyCaret notebooks (Modules 3, 4, 5 and the stock-market examples), always start from **Runtime → Disconnect and delete runtime** and run the install cell first. PyCaret decides which model libraries exist the moment it is imported, so installing after importing leaves models missing.

### Option 2: uv — recommended local setup (Modules 2–5)

The repository root **is** the course environment: one `pyproject.toml`, one `uv.lock`, Python 3.13. `uv` downloads the right Python, creates an isolated `.venv`, and installs the exact package versions recorded in `uv.lock`. It is about 1.3 GB.

New to uv? Start with the [Conda → uv student cheatsheet (PDF)](Platforms%20and%20tools/uv/conda_to_uv_student_cheatsheet.pdf). The [HTML version](Platforms%20and%20tools/uv/conda_to_uv_student_cheatsheet.html) is also available.

#### Step 1: Install uv

On **macOS or Linux**, open Terminal and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On **Windows**, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal, then verify with `uv --version`.

#### Step 2: Download the course repository

```bash
git clone https://github.com/PJalgotrader/Deep_forecasting-USU.git
cd Deep_forecasting-USU
```

If you already cloned the repository, update it with `git pull`.

#### Windows: cloning into Google Drive (recommended if you also use Colab)

Keeping the repository inside your Google Drive folder is handy: your `my_hw/` notebooks are then visible to Colab and to your laptop at the same time. On **macOS** this just works: clone and run `uv sync` inside the Drive folder as usual. On **Windows**, Drive appears as a virtual drive letter (`G:\My Drive`) that cannot handle git's lock files or the thousands of small files in a `.venv`, so two things need to live outside Drive: git's own database and the environment.

Open PowerShell **inside your Google Drive folder** and clone like this:

```powershell
git clone --separate-git-dir "$env:USERPROFILE\df-course.git" https://github.com/PJalgotrader/Deep_forecasting-USU.git
cd Deep_forecasting-USU
```

The course files land in Drive; git's database lands in your home folder, and a small `.git` file in the repository points to it. `git pull`, `git status` and everything else work as usual.

Then, in **every PowerShell session** where you run `uv`, tell uv to build the environment in your home folder instead of inside the Drive folder. Run this line **before** `uv sync` or `uv run`:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE\df-venv"
```

After that, continue with Step 3 exactly as written. If you use VS Code with the registered kernel (below), you only need this line when installing or updating; VS Code finds the kernel on its own.

Do **not** clone into Drive on Windows without `--separate-git-dir`: the clone usually fails midway with `cannot lock ref 'HEAD'` or `Unlink of file ... failed`.

#### Step 3: Install the course environment

From the repository root:

```bash
uv sync
```

Then check it:

```bash
uv run python scripts/check_environment.py
```

If the check ends with **Your course environment is ready**, you are done. It confirms Python 3.13, PyCaret, the forecasting libraries, and trains a small model.

#### Step 4: Start JupyterLab

```bash
uv run jupyter lab
```

Open a notebook and run its cells from top to bottom. You do not need to activate anything or run `pip install`.

#### Use notebooks in VS Code

Register the environment once as a named Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name deep-forecasting --display-name "Python 3.13 (Deep Forecasting)"
```

Then open the repository in VS Code (`code .`, or **File → Open Folder**), open a notebook, click **Select Kernel** in the upper-right corner, and choose **Python 3.13 (Deep Forecasting)**. VS Code remembers the kernel per notebook. To confirm, run `import sys; print(sys.executable)` in a cell: the path should contain the repository's `.venv` (or `df-venv` if the repository is in Google Drive).

#### Updating later

```bash
git pull
uv sync
```

Windows with the repository in Google Drive? Set `UV_PROJECT_ENVIRONMENT` first, as in the Drive section above.

Your own work is not touched by `git pull` as long as it lives in [`my_hw/`](#-your-homework-folder-my_hw).

### Option 3: Conda — supported alternative (Modules 2–5)

If you already use Anaconda or Miniconda:

```bash
conda env create -f python_migration/conda/environment.yml
conda activate df_pycaret
python scripts/check_environment.py
python -m jupyter lab
```

This creates the same Python 3.13 environment. In VS Code, pick **df_pycaret** under **Select Kernel → Python Environments**.

### Modules 6–7: Google Colab

The deep learning notebooks use TensorFlow/Keras and train much faster on a GPU, so the course runs them on Colab. Open the notebook's Colab badge and choose a GPU runtime. There is no local environment to install for these modules.

### Module 8 (optional): Prophet and NeuralProphet

Run it on Colab, or locally in a small environment of its own. NeuralProphet requires Python 3.12 or older and NumPy below 2, which is why it is kept out of the course environment:

```bash
conda create -n df-prophet python=3.11 -y
conda activate df-prophet
python -m pip install prophet neuralprophet jupyterlab
python -m jupyter lab
```

### Why PyCaret is installed as `pycaret-core`

The official `pycaret` 3.3.2 package refuses to import on Python 3.12 or newer, and Google Colab is on Python 3.13. The course uses the community fork `pycaret-core`, which keeps the same API. What broke, what was tested, and the one temporary pin (`statsmodels<0.15`) are explained in [`python_migration/`](python_migration/) and its [tutorial](python_migration/tutorial.html).

### Quick troubleshooting

- **`uv: command not found`** or **`'uv' is not recognized`**: either uv is not installed yet (Step 1) or the terminal was opened before the install. Close and reopen Terminal or PowerShell, then try `uv --version` again.
- **Windows: `git clone` into Google Drive fails** with `cannot lock ref 'HEAD'`, `Invalid argument` or `Unlink of file ... failed`: delete the half-made folder and clone again with `--separate-git-dir` as shown in [Windows: cloning into Google Drive](#windows-cloning-into-google-drive-recommended-if-you-also-use-colab).
- **Two notebooks show as modified right after cloning on Windows** (`Predicting_stock_price_PyCaret.ipynb` and `Predictiong_stock_price_PyCaret.ipynb`): a line-ending artifact, not a real change. Ignore it; `git pull` still works.
- **Wrong Python version**: run `uv run python --version` from the repository root. It should report Python 3.13.
- **Environment feels broken**: delete the `.venv` folder and run `uv sync` again. Nothing is lost; the recipe lives in `pyproject.toml` and `uv.lock`.
- **PyCaret import error** saying *Pycaret only supports python 3.9, 3.10, 3.11*: you installed the old official package. Use `pycaret-core` (see [`python_migration/`](python_migration/)).
- **`Estimator catboost_cds_dt Not Available`** on Colab: the runtime was reused. Runtime → Disconnect and delete runtime, then run the install cell first.
- **Windows: DLL errors when importing packages in a notebook**: the environment was probably built on an Anaconda Python. This repository tells uv to use its own managed Python; delete `.venv` and run `uv sync` again.
- **Windows: `Failed to persist temporary file ... os error 3`**: the folder path is too long. Move the repository to a short path such as `C:\dev\Deep_forecasting-USU`.
- **`code: command not found`**: install the VS Code shell command from the Command Palette, or open the repository with **File → Open Folder**.
- **Local setup is taking too long**: use the notebook's Colab badge instead.

For additional `uv` help, see the official [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/) and [Jupyter integration guide](https://docs.astral.sh/uv/guides/integration/jupyter/).

---

## 📝 Your Homework Folder: `my_hw/`

If you work locally, keep your homework inside a folder named `my_hw` at the repository root. Create it once:

```bash
mkdir my_hw
```

The name has to be exactly `my_hw`. The repository's [`.gitignore`](.gitignore) ignores that folder and everything inside it, which gives you three things:

- **`git pull` stays clean.** Git never looks inside `my_hw/`, so pulling new course material cannot conflict with your work or overwrite it.
- **Your work stays private.** Nothing in `my_hw/` can be committed or pushed by accident, even if you fork the repository.
- **Same environment.** The folder sits inside the repository, so your notebooks use the course `.venv` and the **Python 3.13 (Deep Forecasting)** kernel with no extra setup.

A layout like this works well:

```text
Deep_forecasting-USU/
├── Lectures and codes/
├── data/
└── my_hw/              <- yours, ignored by git
    ├── hw1/
    ├── hw2/
    └── final_project/
```

Two habits to pick up:

- **Copy, don't edit.** To experiment with a lecture notebook, copy it into `my_hw/` and work on the copy. Editing the original is what causes `git pull` conflicts later. From a notebook in `my_hw/hw1/`, the course datasets are at `../../data/`.
- **Back it up yourself.** Because git ignores `my_hw/`, it is not on GitHub and `git pull` will not restore it. Keep a copy in OneDrive, Google Drive, or Box. Deleting `.venv` is safe; deleting the repository folder deletes your homework with it.

Submit homework the way the assignment asks (the repository is not a submission channel). Working on Colab instead? Save your notebooks to Google Drive with **File → Save a copy in Drive**; `my_hw/` only matters for local work.

---

## 📖 Additional Resources

### Video Tutorials
- 📺 [Python Crash Course Playlist](https://www.youtube.com/playlist?list=PL2GWo47BFyUPsqzaOdIdZlAwQmrXkSJxX)
- 📺 [Google Colab Tutorial](https://www.youtube.com/playlist?list=PL2GWo47BFyUOsj5rxrF9s6vRn0HCBEhpW)
- 📺 [PyCaret Time Series](https://youtube.com/playlist?list=PL2GWo47BFyUOqCAj_16yeNspfeM0nfA6q)

### Recommended Reading
- *Forecasting: Principles and Practice* by Hyndman & Athanasopoulos
- *Deep Learning* by Goodfellow, Bengio, and Courville
- Course papers in [`Lectures and codes/`](Lectures%20and%20codes/)

### Useful Links
- [Course GitHub Repository](https://github.com/PJalgotrader/Deep_forecasting-USU)
- [Analytics Solutions Center](https://huntsman.usu.edu/asc/index)
- [Huntsman School of Business](https://huntsman.usu.edu/)

---

## 👨‍🏫 Instructor

**Pedram Jahangiry, PhD, CFA**  
Professional Practice Assistant Professor  
Data Analytics and Information Systems  
Huntsman School of Business, Utah State University

- 📧 Email: pedram.jahangiry@usu.edu
- 🔗 [LinkedIn](https://www.linkedin.com/in/pedram-jahangiry-cfa-5778015a)
- 📺 [YouTube Channel](https://www.youtube.com/channel/UCNDElcuuyX-2pSatVBDpJJQ)
- 🐦 [Twitter/X](https://twitter.com/PedramJahangiry)

**Office Hours**: By appointment

### Background
Dr. Jahangiry brings extensive industry experience from his role as a Research Associate in the Financial Modeling Group at BlackRock NYC. His research focuses on machine learning, deep learning, and time series forecasting applications in finance and business analytics. He mentors students at the Analytics Solutions Center, providing hands-on experience with real corporate analytics projects.

---

## 🤝 Contributing

We welcome contributions from students and the community! Please feel free to:
- Report issues or bugs
- Suggest improvements or new examples
- Share your projects and applications
- Submit pull requests with enhancements

---

## 📄 License

This course material is freely available for educational purposes. All rights reserved by Dr. Pedram Jahangiry and Utah State University.

---

## 🙏 Acknowledgments

Special thanks to:
- All students and contributors who have helped improve this material
- The open-source community for the amazing tools and libraries

---

<div align="center">
  <img src="images/Jahangirylogo.png" width="150" alt="Course Logo">
  
  **Fall 2026 | Utah State University**
  
  *Empowering the next generation of data scientists and forecasting experts*
</div>
