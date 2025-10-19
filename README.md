# Police Search Bias Analysis

This repository presents a full machine learning project investigating patterns in police stop and search data.  
The goal is to explore potential disparities and predictive patterns based on demographics, geography, and search outcomes.

## 🎯 Objective
Use open data from the **Stanford Open Policing Project** to evaluate whether features like driver age, gender, or race influence search outcomes or stop frequency.

## 🧰 Technologies
- Python, Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn (Logistic Regression, Decision Trees, Random Forest, SVM)  
- Tableau for interactive dashboards  
- Jupyter Notebooks for data exploration  

## 🧠 Key Steps
1. **Data Acquisition:** Pulled from Stanford Open Policing dataset.  
2. **Cleaning:** Removed duplicates, handled missing demographic data, normalized search results.  
3. **EDA:** Visualized stop counts, outcomes, and driver demographics by location.  
4. **Modeling:** Predicted the likelihood of a search using ML classification algorithms.  
5. **Ethical Reflection:** Discussed algorithmic fairness, data bias, and social implications.

## 📊 Results
| Model | Accuracy | F1-Score | Notes |
|-------|-----------|----------|-------|
| Logistic Regression | 83% | 0.78 | Baseline performance |
| Decision Tree | 88% | 0.85 | Balanced precision-recall |
| Random Forest | 90% | 0.87 | Best performing model |

## 📈 Visualizations
- Tableau dashboard showing stop and search breakdowns  
- Python plots: race vs. search rate, time-of-day heatmaps, model evaluation charts  

## 🗂 Repository Structure

## 🧩 Learning Outcomes
- Hands-on experience applying ML to real-world social data  
- Enhanced understanding of fairness metrics and bias in algorithms  
- Strengthened skills in ethical data interpretation and visualization  

---

