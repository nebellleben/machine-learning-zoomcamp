# 🎓 Midterm Project — Student Performance Prediction

Note: Jump right to [here](#72-execution-instructions) for deployment instructions.

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

### 6.3 Optional: Cloud Deployment (Skipped)

## 7. Reproducibility and Environment Management

### 7.1 Dependency File

* Dependencies have been built into the uv files (`pyproject.toml` and `uv.lock`)
* This is how it was done (you don't have to execute the following code)
```bash
uv init
uv add scikit-learn fastapi uvicorn xgboost pandas numpy matplotlib seaborn
uv add --dev requests
```

### 7.2 Execution Instructions

1. Clone this repository `git pull https://github.com/nebellleben/machine-learning-zoomcamp/`; if it doesn't work please go to the project page to download the zip file manually
2. Go to the sub-directory `midterm-project` which contains the project files
3. In case you didn't have Docker installed please install it
4. Run Docker Image with the following commands in terminal:
```bash
docker build -t student-score-prediction .
docker run -it --rm -p 9696:9696 student-score-prediction
```
5. Open the browser and access the location [http://localhost:9696/docs](http://localhost:9696/docs)
6. You can test out the prediction with the following dummy data:
```
{
    "gender": "female",
    "race/ethnicity":"group B",
    "parental level of education":"master's degree",
    "lunch":"standard",
    "test preparation course":"none"
}
```
7. The predicted score should be shown as below:
```
{
  "math score": 66.417124605768,
  "reading score": 74.56104490095517,
  "writing score": 74.16979922907296
}
```

## 8. Reflection and Next Steps

* The dataset is not large - only 1000 entries
* I believe that if the dataset is bigger and if there are more features, more accurate predications can be made
* Also surprisingly more sophisticated model didn't give better results, probably related to the same problem of limited size of dataset and number of features
* During EDA the following insights were observed:
 * Male performs better at math and female performs better at reading and writing, on average
 * Reading and writing scores are strongly correlated
 * Social background does have positive correlation on the test score, ranking from lowest score to highest score: group A < B < C < D < E
 * Doing test preps and having lunch do improve scores
 * Parents' education background does have positive correlation with test scores
* My biggest difficulties are with deployment, how thanks to Alexey I had hands on experience with using `uv` and `Docker`


## 9. References

* [Machine Learning Zoomcamp - Project Guidelines](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects)
* [Kaggle Dataset: Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
