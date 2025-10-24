#  Police Search Bias Prediction
**Using Machine Learning to Analyze Racial and Situational Bias in Police Stop Data**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Library-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

###  Project Overview
This project investigates  **patterns in police stop data** and uncover potential bias in police search decisions using open-source stop data. By applying machine learning to real-world law enforcement data, it aims to reveal correlations between **stop reasons, race, vehicle searches, and precincts**, highlighting both predictive power and fairness concerns.


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

After cleaning with `dropna()` and encoding via `pd.get_dummies()`, the data was split into **training (80%)** and **testing (20%)** sets.

---

###  Workflow & Methodlogy

####  Step 1: Data Cleaning & Encoding
- Removed missing values, encoded categorical variables using **`pd.get_dummies()`**  
- Normalized features and dropped incomplete rows
- Verified column types and balanced the dataset for fair modeling.
- Performed visual EDA with `Seaborn` and `Matplotlib`

####  Step 2: Feature Engineering
- Predictor columns: `reason`, `vehicleSearch`, `precinct`, etc.  
- Target variable: `personSearch_YES` (binary classification)

####  Step 3: Exploratory Data Analysis (EDA)
- Visualized stop distributions and racial disparities using **Seaborn**.
- Example plots:
  - Count of `personSearch_YES` by problem type.
  - Precinct frequency histograms.
  - Comparison by racial subgroups (`White`, `Asian`, `Black`).

####  Step 4: Model Training
Implemented multiple models using **Scikit-learn**:

| Model | Description |
|--------|-------------|
|  K-Nearest Neighbors (k=3) | Measures similarity between observations |
|  Decision Tree | Visualizes decision logic and features |
|  Random Forest | Ensemble averaging for stronger prediction |
|  SVM | Linear, RBF, Polynomial, and Sigmoid kernels |
|  Logistic Regression | Probabilistic binary classifier |

####  Step 5: Evaluation
- Confusion matrices & accuracy scores for all models.  
- Compared kernel performance for SVM.  
- Visualized **Logistic Regression results** with a custom scatter plot and confusion matrix heatmap.


---



###  Results

| Model | Accuracy (Approx.) | Key Insight |
|--------|-------------------|--------------|
| **KNN (k=3)** | ~0.80 | Sensitive to feature scaling |
| **Decision Tree** | ~0.83 | Easily interpretable, may overfit |
| **Random Forest** | ~0.85 | Best overall balance and generalization |
| **SVM (Linear/RBF)** | ~0.82 | Linear kernel performed most consistently |
| **Logistic Regression** | ~0.84 | Clear interpretability and stable results |

**Confusion matrices** confirm consistent classification between “Searched” and “Not Searched” classes.

</details>



---

###  Key Visuals
- Distribution of searches by race and problem type  
- Correlation matrix for predictive features  
- Confusion matrices for all classifiers  
- Scatter plots for logistic regression predictions  

---


###  Insights
- *Vehicle searches* and *stop reasons* emerged as the strongest predictors of whether a person was searched.  
- Certain precincts and violation types displayed disproportionately higher search rates, indicating potential systemic bias.  
- Logistic Regression and Random Forest models both confirmed consistent patterns aligning with known real-world enforcement disparities.  
- Visualizing the confusion matrices and feature importances helped connect quantitative model results with social impact interpretation.


---

###  Tech Stack



</details>

**Languages:** Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)  
**Tools:** Jupyter Notebook, Graphviz, GitHub  
**Techniques:** Data encoding, feature engineering, model benchmarking, and confusion matrix visualization  

**Libraries Used:**
- `Pandas` – data manipulation and preprocessing  
- `Seaborn` / `Matplotlib` – data visualization  
- `Scikit-learn` – model training and evaluation  
- `Graphviz` – decision tree visualization


---

###  Future Work



</details>

Add **feature scaling** and **cross-validation** for higher accuracy

Integrate Explainable AI (SHAP, LIME) for interpretability.

Build a Flask/Streamlit dashboard for interactive data visualization and real time analysis

Expand dataset to include multi-state comparisons to expand predictive content.

Incorporate fairness metrics (e.g., Equalized Odds, Disparate Impact).


---

*** Repository Structure


├── Police_Search_Bias.ipynb
├── data/
│   └── Police_stop_data1.csv
├── visuals/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
└── README.md



---

##  Author  
**Saanvi ([@SalmaLilad](https://github.com/SalmaLilad))**  
Exploring fairness, ethics, and transparency in real-world machine learning applications.

---

>  *“Data science is not just about prediction — it’s about understanding the patterns that shape human impact.”*

---
