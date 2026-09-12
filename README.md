# 🔍 Data Forensics AI

### AI-Powered Data Quality & Anomaly Detection Platform

**Data Forensics AI** is a Flask-based Data Science application that automatically investigates datasets for quality issues, suspicious records, and inconsistencies. It combines data analysis, automated cleaning, visualization, and machine learning to produce a clear forensic assessment of the dataset.

## 🚀 Key Features

- 📊 Automated dataset profiling and quality analysis
- 🔎 Missing, duplicate, and invalid value detection
- 📉 Outlier detection using **IQR**
- 🤖 Anomaly detection using **Isolation Forest**
- 🧹 Automated data cleaning and cleaned CSV generation
- 📈 Correlation analysis and data visualizations
- 📋 Automated **Forensic Quality Score**
- 💡 Actionable data-quality recommendations
- 🗄️ MySQL storage for forensic analysis results
- 📑 Web-based forensic results dashboard

## 🧠 Machine Learning

The system uses **Isolation Forest**, an unsupervised machine learning algorithm, to identify unusual records without requiring labeled training data.

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Scaling
   ↓
Isolation Forest
   ↓
Anomaly Detection
   ↓
Anomaly Score
```

## 🛠️ Tech Stack

**Python | Flask | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | MySQL | HTML/CSS**

## 🔄 Workflow

```text
CSV Upload
    ↓
Data Quality Analysis
    ↓
Validation & Cleaning
    ↓
Outlier Detection
    ↓
ML Anomaly Detection
    ↓
Forensic Score
    ↓
Recommendations & Report
```

## 🎯 Use Cases

- Data Quality Auditing
- Dataset Preprocessing
- Data Cleaning
- Anomaly Investigation
- Machine Learning Data Preparation
- Data Reliability Assessment

## ⚙️ Run Locally

```bash
git clone https://github.com/srushti-dsjourney/Data-Forensics-AI.git
cd Data-Forensics-AI
pip install -r requirements.txt
python app.py
```

Create a `.env` file with your MySQL configuration before running the application.

## 👩‍💻 Author

**Srushti Pawar**  
BSc Data Science

**Focus:** Data Science • Python • SQL • Machine Learning • Data Analytics

---

⭐ **Portfolio Project | Data Science & Machine Learning**
