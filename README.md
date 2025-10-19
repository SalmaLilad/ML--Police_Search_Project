# 🚓 Police Search Bias Prediction
**Using Machine Learning to Analyze Racial and Situational Bias in Police Stop Data**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Library-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

###  Project Overview
This project investigates potential bias in police search decisions using open-source stop data. Through **data visualization** and **machine learning models**, it predicts whether a *person was searched* based on demographics, violation type, and stop reason.

Built as part of the University of Minnesota 2025 Basic Machine Learning Summer Camp **Capstone Project**, it explores fairness and real-world applications of AI in social data.

---

###  Dataset
- **Source:** [MLCamp2025 Police Stop Dataset](https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/Police_stop_data1.csv)
- **Size:** ~20,000 records, 15+ categorical variables
- **Key Variables:**
  - `preRace` – perceived race
  - `reason` – reason for stop
  - `problem` – violation type
  - `vehicleSearch`, `personSearch` – binary outcomes
  - `gender`, `callDisposition`, `citationIssued`, `precinct`

---

###  Methodology

#### 1️⃣ Data Cleaning & Preparation
- Removed missing values, encoded categorical variables using **`pd.get_dummies()`**  
- Normalized features and dropped incomplete rows  
- Performed visual EDA with `Seaborn` and `Matplotlib`

#### 2️⃣ Feature Engineering
- Predictor columns: `reason`, `vehicleSearch`, `precinct`, etc.  
- Target variable: `personSearch_YES` (binary classification)

#### 3️⃣ Model Comparison
| Model | Description | Accuracy |
|:------|:-------------|:----------:|
| **KNN** | Baseline classification | ~65% |
| **Decision Tree** | Interpretable model (visualized via Graphviz) | ~70% |
| **Random Forest** | Ensemble model with improved stability | ~73% |
| **SVM (Linear, RBF, Poly, Sigmoid)** | Tested multiple kernels | 68–75% |
| **Logistic Regression** | Probabilistic baseline | ~72% |

---

###  Key Visuals
- Distribution of searches by race and problem type  
- Correlation matrix for predictive features  
- Confusion matrices for all classifiers  
- Scatter plots for logistic regression predictions  

> Example output :

```python
sns.countplot(x='personSearch_YES', data=police_df_new)
plt.title("Distribution of Person Searches")
plt.show()


---

###  Insights

Vehicle searches and stop reasons are strongest predictors.

Some precincts and violation types have disproportionately high search rates.

Machine learning revealed consistent patterns that align with known bias indicators.


---

###  Tech Stack

Languages: Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)
Tools: Jupyter Notebook, Graphviz, GitHub
Techniques: Data encoding, model benchmarking, confusion matrix visualization


---

###  Future Work

Integrate Explainable AI (SHAP, LIME) for interpretability.

Build a Flask/Streamlit dashboard for interactive data visualization.

Expand dataset to include multi-state comparisons.

Incorporate fairness metrics (e.g., Equalized Odds, Disparate Impact).


---

###  Citation
Lilad, S. (2025). Police Search Bias Prediction Using Machine Learning.

---

*** Repository Structure

├── Police_Search_Bias.ipynb
├── data/
│   └── Police_stop_data1.csv
├── visuals/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
└── README.md



