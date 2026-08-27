![DF cover](https://user-images.githubusercontent.com/19335954/210499919-b5000dda-b46c-42b9-b274-fe06116c8260.png)

# Deep Forecasting - Fall 2026
### Advanced Time Series Analysis and Forecasting with Deep Learning
**Utah State University | Huntsman School of Business**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/PJalgotrader/Deep_forecasting-USU)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

> [!IMPORTANT]
> You are viewing the experimental `uv_test` branch. It provides three setup choices for comparison: Google Colab, `uv`, and Conda. The [`main` branch](https://github.com/PJalgotrader/Deep_forecasting-USU/tree/main) remains unchanged.

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
- **Software**: Google account for Colab access (no local installation required)

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
- **[PyCaret](Platforms%20and%20tools/PyCaret/)**: AutoML for time series
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

Choose **one** of the following paths. If you are unsure, use Google Colab. It requires no installation.

### Option 1: Google Colab — easiest

1. Open a notebook on GitHub.
2. Click its **Open in Colab** badge.
3. Sign in with your Google account.
4. Run cells from top to bottom with `Shift+Enter`.

Colab is recommended when you want GPU access or cannot install software on your computer. Package-install cells inside notebooks are intended for Colab.

### Option 2: uv — recommended local setup

`uv` installs the correct Python version, creates an isolated environment, and installs the exact package versions recorded in `uv.lock`.

#### Step 1: Install uv

On **macOS or Linux**, open Terminal and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal, then verify the installation:

```bash
uv --version
```

On **Windows**, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then run `uv --version`.

#### Step 2: Download this test branch

```bash
git clone --branch uv_test https://github.com/PJalgotrader/Deep_forecasting-USU.git
cd Deep_forecasting-USU
```

If you already cloned the repository, switch to the test branch with:

```bash
git switch uv_test
git pull
```

#### Step 3: Install the course environment

```bash
uv sync
```

The first installation can take several minutes because the course uses TensorFlow, PyTorch, and forecasting libraries. Future synchronizations are usually much faster.

Confirm that the correct Python is being used:

```bash
uv run python --version
```

The result should begin with `Python 3.11`.

Run the course environment check:

```bash
uv run python scripts/check_environment.py
```

If the check ends with **Your main course environment is ready**, the installation worked.

#### Step 4: Start JupyterLab

```bash
uv run jupyter lab
```

Your browser will open JupyterLab. Open a notebook and run its cells from top to bottom. You do not need to activate the environment or run `pip install`.

#### PyCaret notebooks

PyCaret uses a separate environment because its full installation requires older versions of some forecasting packages. When the course reaches a PyCaret notebook, run:

```bash
uv sync --project environments/pycaret
uv run --project environments/pycaret python environments/pycaret/check_environment.py
uv run --project environments/pycaret jupyter lab
```

If the check ends with **Your PyCaret environment is ready**, the installation worked. This environment uses Python 3.10 and does not change the main course environment. Exit JupyterLab before switching between the two environments.

#### Updating later

```bash
git pull
uv sync
```

### Option 3: Conda — supported alternative

Use this path if you already have Anaconda or Miniconda and prefer Conda. Run these commands from the repository folder.

#### Main course environment

```bash
conda create -n deep-forecasting python=3.11 -y
conda activate deep-forecasting
python -m pip install -r requirements.txt
python -m jupyter lab
```

#### PyCaret environment

```bash
conda create -n deep-forecasting-pycaret python=3.10 -y
conda activate deep-forecasting-pycaret
python -m pip install -r environments/pycaret/requirements.txt
python -m jupyter lab
```

Use only one active Conda environment at a time. Run `conda deactivate` before switching environments.

### Quick troubleshooting

- **`uv: command not found`**: close and reopen Terminal or PowerShell, then try `uv --version` again.
- **Wrong Python version**: run `uv run python --version` from the repository root. It should report Python 3.11.
- **PyCaret import error**: close JupyterLab and restart it using the two PyCaret commands above.
- **Notebook uses the wrong kernel**: restart JupyterLab from the correct environment rather than installing packages inside the notebook.
- **Local setup is taking too long**: use the notebook's Colab badge instead.

For additional `uv` help, see the official [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/) and [Jupyter integration guide](https://docs.astral.sh/uv/guides/integration/jupyter/).

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
- The Huntsman School of Business for supporting this course
- The Analytics Solutions Center team
- All students and contributors who have helped improve this material
- The open-source community for the amazing tools and libraries

---

<div align="center">
  <img src="images/Jahangirylogo.png" width="150" alt="Course Logo">
  
  **Fall 2026 | Utah State University**
  
  *Empowering the next generation of data scientists and forecasting experts*
</div>
