# 🎓 Midterm Project — Student Performance Prediction

## 1. Project Overview

### 1.1 Overview and Problem Statement

The goal of this project is to predict **students' academic performance** — specifically their **math, reading, and writing scores** — based on demographic and educational background factors such as gender, parental education level, lunch type, and test preparation course.

This predictive model can help educators identify students who may need additional support in specific subjects.

### 1.2 Dataset

* **Source:** [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
* **Description:** The dataset contains demographic and test information of students, including:

  * Gender
  * Race/Ethnicity
  * Parental level of education
  * Lunch type
  * Test preparation course
  * Math score (target variable)
  * Reading score (target variable)
  * Writing score (target variable)


## 2. Data Preparation and Exploratory Data Analysis (EDA)

### 2.1 Import Libraries and Dataset

* Import essential libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
* Load dataset and display first few rows using `head()`

### 2.2 Basic Inspection

* Check dataset shape, column names, and data types
* Look for missing values and duplicates
* Identify numerical and categorical features

### 2.3 Exploratory Visualization

* Histograms for score distributions
* Countplots for categorical variables (e.g., gender, parental education)
* Heatmap and Pairplots to explore target interactions

### 2.4 Feature Insights

* Discuss key trends:

  * Gender differences in performance
  * Impact of test preparation course
  * Parental education vs student performance

## 3. Data Preprocessing

### 3.1 Feature-Target Separation

* Define **X (input features)** and **y (targets)** — math, reading, and writing scores.

### 3.2 Train/Validation/Test Split

* Split dataset into training (60%), validation (20%), and test (20%).

### 3.3 Encoding Categorical Variables

* Apply one-hot encoding and transform the variables with `DataVectorizer`.

## 4. Model Training and Evaluation

### 4.1 Baseline Model

* Simple model such as mean prediction or linear regression baseline.

### 4.2 Linear Regression (Lasso)

* Train Linear Regression (Lasso)
* Tune hyperparameter `alpha` using grid search
* Evaluate with RMSE metrics on validation set

### 4.3 Decision Tree Regressor

* Train Decision Tree model
* Tune `max_depth` and `min_samples_split`
* Evaluate and visualize feature importance

### 4.4 Random Forest Regressor

* Train Random Forest model
* Tune `n_estimators` and `max_depth`
* Compare validation performance and interpret feature importance

### 4.5 XGBoost Regressor

* Train and tune XGBoost model
* Tune `eta`
* Evaluate metrics and analyze feature importance

### 4.6 Model Comparison

* Create summary table comparing all models on validation data (RMSE)
* Select the **best-performing model**

## 5. Model Export and Scripting

### 5.1 Export Training Logic

* Move final training pipeline to `train.py`
* Save trained model (e.g., using `joblib` or `pickle`)

### 5.2 Prediction Script

* Implement `predict.py` to load model and perform inference on new data.


## 6. Model Deployment (Local)

### 6.1 Simple Web Service

* Create Flask or FastAPI endpoint to serve predictions (e.g., `/predict`)
* Test locally

### 6.2 Dockerization

* Write Dockerfile
* Run and test container locally (`docker build`, `docker run`)

### 6.3 Optional: Cloud Deployment

* (Optional) Deploy to cloud platform (e.g., Render, Railway, or Hugging Face Spaces)
* Include **URL** or **video screenshot** in README

---

## 7. Reproducibility and Environment Management

### 7.1 Dependency File

* Include `requirements.txt` or `Pipfile` listing all dependencies.

### 7.2 Virtual Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 7.3 Execution Instructions

1. Clone this repository
2. Install dependencies
3. Run `train.py` to train the model
4. Run `predict.py` to test predictions
5. Start Flask/FastAPI app via `python predict.py`
6. (Optional) Build Docker image and deploy

---
## 8. Reflection and Next Steps

* Summarize findings and model performance
* Discuss key features influencing student performance
* Outline potential improvements (e.g., handling outliers, adding new data)
* Reflect on lessons learned from the project

---

## 9. References

* [Machine Learning Zoomcamp - Project Guidelines](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects)
* [Kaggle Dataset: Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
