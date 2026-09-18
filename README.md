# 🩺 Early Sepsis Risk Predictor

### Machine Learning for Early Detection of Sepsis from ICU Time-Series Data

An end-to-end machine learning project for **early sepsis risk prediction** using ICU time-series data from the **PhysioNet/CinC Challenge 2019** dataset.

The project explores how routinely collected physiological and clinical measurements can be used to identify patients at increased risk of developing sepsis, with the goal of supporting **earlier clinical intervention**, particularly in resource-limited healthcare settings.

---

## 👥 Authors

**Atta Ullah**
Bioinformatics Graduate | Machine Learning & Computational Biology

**Mehak Jabeen**
Bioinformatics / Computational Biology

Developed during our work at **Genyx**.

---

## 🎯 Project Overview

Sepsis is a life-threatening medical condition caused by a dysregulated response to infection. Early recognition is critical because delayed treatment can substantially increase the risk of organ dysfunction and death.

This project investigates whether machine learning can learn clinically meaningful patterns from ICU time-series data and provide an **early warning of sepsis risk**.

### Key objectives

* Predict sepsis risk using physiological and clinical measurements.
* Investigate patterns preceding sepsis onset.
* Compare machine-learning approaches for early prediction.
* Address class imbalance in sepsis data.
* Evaluate model performance using clinically relevant metrics.
* Develop a simple interactive prediction interface.
* Explore the potential application of ML in resource-limited hospitals.

---

## 📊 Dataset

The project uses the:

**PhysioNet/Computing in Cardiology Challenge 2019**

The dataset contains ICU patient records represented as hourly time-series observations.

The data include measurements such as:

* Heart rate
* Respiratory rate
* Oxygen saturation
* Temperature
* Blood pressure
* Laboratory measurements
* Demographic information
* Other physiological variables

### Data Sources

* **Set A**
* **Set B**

The challenge data were used to develop and evaluate machine-learning models for early sepsis prediction.

> **Important:** The original dataset is subject to the PhysioNet data-use requirements. Users should obtain the dataset directly from PhysioNet and follow its applicable terms and credentialing requirements.

---

## 🧬 Prediction Target

The project uses the **6-hour sepsis prediction target (`Target_6h`)**.

The model aims to determine whether a patient is at risk of developing sepsis within the specified prediction window based on information available before the prediction time.

### Binary classification

| Label | Meaning                              |
| ----- | ------------------------------------ |
| `0`   | No sepsis within prediction window   |
| `1`   | Sepsis risk within prediction window |

---

## 🧠 Modeling Strategy

The project evaluates machine-learning methods for binary sepsis prediction.

Potential models include:

* Logistic Regression
* Random Forest
* Support Vector Machine
* XGBoost
* LightGBM
* Other tree-based ensemble models

The final model selection is based on multiple evaluation metrics rather than accuracy alone.

---

## ⚖️ Class Imbalance

Sepsis prediction is an inherently imbalanced classification problem because the number of non-sepsis observations is substantially larger than the number of sepsis observations.

Therefore, the project considers:

* Class weighting
* Resampling strategies
* Precision-Recall analysis
* F1-score
* Matthews Correlation Coefficient (MCC)
* Area Under the Precision-Recall Curve (AUPRC)

This is important because a model can achieve high accuracy while performing poorly on the minority sepsis class.

---

## 📈 Model Evaluation

The models are evaluated using:

| Metric               | Purpose                                         |
| -------------------- | ----------------------------------------------- |
| Accuracy             | Overall classification performance              |
| Precision            | Reliability of positive predictions             |
| Recall / Sensitivity | Ability to identify sepsis cases                |
| Specificity          | Ability to identify non-sepsis cases            |
| F1-score             | Balance between precision and recall            |
| AUROC                | Overall discrimination                          |
| AUPRC                | Performance under class imbalance               |
| MCC                  | Robust evaluation for imbalanced classification |

### Why AUPRC and MCC?

For highly imbalanced medical datasets, AUROC and accuracy alone may not adequately describe minority-class performance.

Therefore, **AUPRC and MCC** are emphasized alongside sensitivity, specificity, and AUROC.

---

## 🖥️ Interactive Application

A web-based version of the project was developed using **Streamlit**.

### 🚀 Live Demo

**Early Sepsis Risk Predictor**

https://sepsis-detection-app-4w2abxbdwswgzpd4phusy7.streamlit.app/

The application provides an interactive interface for entering clinical variables and obtaining a machine-learning-based sepsis risk prediction.

---

## 🧪 Two Prediction Modes

The application explores two feature configurations:

### 1. Full-Data Model

Uses a broader set of available clinical and physiological variables.

### 2. Vitals-Only Model

Uses primarily routinely available vital-sign information.

The vitals-only approach is particularly relevant to **resource-limited healthcare environments**, where advanced laboratory testing may not always be immediately available.

---

## 💻 Technologies

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest
* XGBoost
* Support Vector Machine
* Logistic Regression

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---


## 🔄 Reproducibility

To reproduce the analysis:

1. Obtain the PhysioNet/CinC 2019 dataset.
2. Place the required files in the appropriate data directory.
3. Run preprocessing.
4. Perform feature engineering.
5. Train the machine-learning models.
6. Evaluate the models using the provided metrics.
7. Save the trained model.
8. Launch the Streamlit application.

---


### PhysioNet/CinC Challenge 2019

Reyna, M. A., et al.
**Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.**

---

## 🤝 Collaboration

Contributions, research collaborations, suggestions, and improvements are welcome.

If you are interested in **Bioinformatics, Machine Learning, Computational Biology, Healthcare AI, or Clinical Data Science**, feel free to connect and collaborate.

---


### 🩺 Early Detection • Machine Learning • Healthcare AI • Sepsis Prediction

**Built by Atta Ullah & Mehak Jabeen**

