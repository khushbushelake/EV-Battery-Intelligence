# 🔋 EV Battery Intelligence System

### AI-Powered Lithium-Ion Battery State of Charge (SOC) Estimation

An end-to-end deep learning research project that compares **LSTM, GRU, and BiLSTM** architectures for estimating the State of Charge (SOC) of lithium-ion batteries using time-series data.

## 📌 Project Overview

The EV Battery Intelligence System analyzes experimental battery data and evaluates deep learning models to estimate battery SOC under different temperature conditions.

The project includes data preprocessing, exploratory data analysis, sequence generation, model training, independent test evaluation, SOC prediction analysis, and an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze lithium-ion battery time-series data.
- Develop LSTM, GRU, and BiLSTM models.
- Compare model performance using MAE, RMSE, and R².
- Evaluate models on independent test datasets.
- Analyze SOC predictions across different temperatures.
- Present results through an interactive dashboard and technical report.

## 📊 Dataset

**Dataset:** LG 18650HG2 Lithium-Ion Battery Dataset

| Parameter | Details |
|---|---|
| Training observations | 669,956 |
| Input features | 5 prepared features |
| Sequence length | 30 time steps |
| Sequence stride | 5 |
| Prediction target | Battery State of Charge |
| Test temperatures | -10°C, 0°C, 10°C, 25°C |

The prepared dataset was used for model development and evaluation.

## 🧠 Deep Learning Models

Three recurrent neural network architectures were developed:

- **LSTM:** Long Short-Term Memory
- **GRU:** Gated Recurrent Unit
- **BiLSTM:** Bidirectional Long Short-Term Memory

### Model Architecture

The models use a recurrent layer with 64 units, followed by dropout, a dense layer with 32 neurons, and a single output neuron.

Training configuration:
- Optimizer: Adam
- Loss function: Mean Squared Error
- Evaluation metrics: MAE
- Maximum epochs: 5
- Early stopping: Patience of 2 epochs

## 📈 Independent Test Results

The models were evaluated on four temperature-specific test datasets.

| Model | Average MAE | Average RMSE | Average R² |
|---|---:|---:|---:|
| **BiLSTM** | **0.01842** | **0.02435** | **0.99021** |
| LSTM | 0.02495 | 0.03214 | 0.98347 |
| GRU | 0.03976 | 0.04711 | 0.96782 |

*Note: These are unweighted arithmetic averages of the four temperature-specific test metrics.*

### Key Finding

BiLSTM achieved the strongest overall independent test performance among the three evaluated architectures.

A total of **34,703 sampled test predictions** were generated across the four temperature conditions.

## 🖥️ Interactive Dashboard

The Streamlit dashboard includes seven sections:

1. Project Overview
2. Battery Data Analysis
3. Model Comparison
4. Independent Test Evaluation
5. SOC Prediction
6. Model Architecture
7. Technical Report

The SOC Prediction section allows exploration of saved experimental predictions, including actual SOC, predicted SOC, and absolute error.

The dashboard is an experimental analysis interface. It is not connected to a live electric vehicle battery.

## 🛠️ Technology Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- ReportLab
- Google Colab
- Visual Studio Code
- Git and GitHub

## 📁 Project Structure

```text
EV-Battery-Intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── lstm_battery_soc.keras
│   ├── gru_battery_soc.keras
│   └── bilstm_battery_soc.keras
│
├── notebooks/
│   └── 01_dataset_exploration.py
│
├── reports/
│   ├── EDA visualizations
│   ├── Model comparison results
│   ├── Independent test results
│   ├── SOC prediction results
│   └── Technical evaluation report
│
├── src/
│   └── prepare_data.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/khushbushelake/EV-Battery-Intelligence.git
```

### 2. Navigate to the project directory

```bash
cd EV-Battery-Intelligence
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app/app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

## 📄 Technical Report

The project includes a professional technical evaluation report covering:

- Executive summary
- Dataset and system overview
- Model configurations
- Validation performance
- Independent test evaluation
- SOC prediction analysis
- Performance visualizations
- Key findings and limitations
- Conclusion

The report is available through the dashboard's Technical Report section.

## ⚠️ Limitations

- Evaluation is based on the prepared LG 18650HG2 experimental dataset.
- Testing covers four specified temperature conditions.
- Results do not establish generalization across all battery chemistries, battery ages, or vehicle operating environments.
- The dashboard analyzes saved experimental predictions and is not a live battery monitoring system.
- Further evaluation on additional cells, cycles, and real-world measurements is needed before practical deployment.

## 🔮 Future Improvements

- Evaluate additional battery datasets and operating conditions.
- Explore real-time battery data integration.
- Improve model generalization across battery cycles.
- Investigate model optimization for edge deployment.
- Extend predictive analysis to battery health and degradation.

## 👩‍💻 Developer

**Khushbu Uttam Shelake**

B.Tech – Computer Science and Engineering (Data Science)

JSS Academy of Technical Education, Noida

- GitHub: [khushbushelake](https://github.com/khushbushelake)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/khushbu-uttam-shelake-aaa4b7372)

---

*Developed as an academic deep learning research project.*

