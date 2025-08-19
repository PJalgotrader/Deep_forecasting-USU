![DF cover](https://user-images.githubusercontent.com/19335954/210499919-b5000dda-b46c-42b9-b274-fe06116c8260.png)

# Deep Forecasting - Fall 2025
### Advanced Time Series Analysis and Forecasting with Deep Learning
**Utah State University | Huntsman School of Business**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/PJalgotrader/Deep_forecasting-USU)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 📚 Course Overview

This comprehensive course introduces students to state-of-the-art time series forecasting techniques, progressing from classical statistical methods to advanced deep learning architectures. Students will gain hands-on experience with real-world forecasting problems using industry-standard tools and frameworks.

### 🎯 Learning Objectives

Upon completion of this course, students will be able to:
- Master fundamental time series concepts and decomposition techniques
- Implement classical forecasting methods (ETS, ARIMA/SARIMA)
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
- **Resources**: [Lecture Slides](Lectures%20and%20codes/Module%201-%20Demystifying%20timeseries%20data%20and%20modeling/)

### Module 2: Setting up Deep Forecasting Environment
- Python environment configuration
- Essential libraries and tools
- Google Colab setup and best practices
- **Lab**: [Environment Setup Notebook](Lectures%20and%20codes/Module%202-%20Setting%20up%20DF%20environment/module2_ts_basics.ipynb)
- **Extra**: [Python Crash Course](Lectures%20and%20codes/Module%202-%20Setting%20up%20DF%20environment/Python_Crash_course_2020/)

### Module 3: Exponential Smoothing Methods
- Simple, Holt's, and Holt-Winters methods
- ETS (Error, Trend, Seasonal) models
- Model selection and validation
- **Lab**: [Exponential Smoothing Implementation](Lectures%20and%20codes/Module%203-%20Exponential%20Smoothing/Module3-exponential_smoothing_ETS.ipynb)

### Module 4: ARIMA Models
- AR, MA, and ARMA processes
- ARIMA and seasonal ARIMA (SARIMA)
- Box-Jenkins methodology
- **Lab**: [ARIMA Modeling](Lectures%20and%20codes/Module%204-%20ARIMA/Module4-ARIMA.ipynb)

### Module 5: Machine Learning for Time Series
- Feature engineering for time series
- Tree-based methods (Random Forest, XGBoost, LightGBM)
- Cross-validation strategies
- **Lab**: [ML Time Series Forecasting](Lectures%20and%20codes/Module%205-%20Machine%20Learning%20Forecasting/Module5-ML_timesereis.ipynb)

### Module 6: Deep Neural Networks
- Feedforward networks for time series
- Backpropagation and optimization
- TensorFlow/Keras implementation
- **Lab**: [DNN for Time Series](Lectures%20and%20codes/Module%206-%20Deep%20Neural%20Networks/Module6_UnivariateTS_DNN.ipynb)

### Module 7: Deep Sequence Modeling
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM) networks
- Bidirectional and stacked architectures
- **Lab**: [RNN/LSTM Implementation](Lectures%20and%20codes/Module%207-%20Deep%20Sequence%20Modeling/Module7_Univariate_RNN-LSTM.ipynb)

### Module 8: Prophet and NeuralProphet
- Forecasting at scale
- Handling seasonality and holidays
- Uncertainty quantification
- **Lab**: [Prophet Tutorial](Lectures%20and%20codes/Module%208-%20Prophet%20and%20NeuralProphet/Module8_prophet_basics.ipynb)

---

## 🛠️ Tools and Platforms

### Primary Frameworks
- **[PyCaret](Platforms%20and%20tools/PyCaret/)**: AutoML for time series
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

### Option 1: Google Colab (Recommended)
1. Click on any notebook's "Open in Colab" button
2. Sign in with your Google account
3. Run cells sequentially (Shift+Enter)

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/PJalgotrader/Deep_forecasting-USU.git
cd Deep_forecasting-USU

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

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

## 📝 Course Policies

### Grading Components
- Homework Assignments: 30%
- Midterm Project: 25%
- Final Project: 35%
- Class Participation: 10%

### Academic Integrity
All submitted work must be original. Collaboration is encouraged, but direct code copying is prohibited. Please cite all sources and acknowledge any assistance received.

### Late Policy
Late submissions incur a 10% penalty per day unless prior arrangements are made.

---

## 🤝 Contributing

We welcome contributions from students and the community! Please feel free to:
- Report issues or bugs
- Suggest improvements or new examples
- Share your projects and applications
- Submit pull requests with enhancements

---

## 📄 License

This course material is available under the MIT License. See [LICENSE](LICENSE) file for details.

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
  
  **Fall 2025 | Utah State University**
  
  *Empowering the next generation of data scientists and forecasting experts*
</div>