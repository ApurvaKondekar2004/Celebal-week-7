#  Titanic Survival Predictor

This project is a web application built using **Streamlit** that predicts whether a passenger on the Titanic would survive or not, based on input features like age, class, fare, etc. The model used is a tuned **XGBoost classifier** trained on the Titanic dataset.

---

# Features

- Streamlit-powered interactive web interface
- Real-time prediction: Survived / Not Survived
- Probability bar chart visualization
- Clean preprocessing pipeline with:
  - Median imputation for age
  - Mode imputation for embarked/cabin
  - Deck extraction from cabin
  - Standard scaling and encoding
- Hyperparameter tuning using **GridSearchCV** and **RandomizedSearchCV**
- Easily deployable via **Streamlit Cloud**



#  Live Demo

 Click to try the app:  
[ Streamlit App Link](https://celebal-week-7-kggdvyhftky5qnajlbxuvy.streamlit.app/)  




##  Dataset

- Source: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Features used:  
  `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Deck` (extracted from Cabin)



#  Model Training

- Preprocessing with `ColumnTransformer` using:
  - `StandardScaler` for numerical features (`Age`, `Fare`)
  - `OneHotEncoder` for categorical features (`Embarked`, `Deck`)
  - `OrdinalEncoder` for binary feature (`Sex`)
- Trained on multiple models (Logistic Regression, SVM, KNN, etc.)
- Final model: **XGBoostClassifier** with best GridSearchCV hyperparameters:
  - `learning_rate`: 0.2
  - `max_depth`: 3
  - `n_estimators`: 50
  - `subsample`: 0.8
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
