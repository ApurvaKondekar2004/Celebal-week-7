import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("xgb_titanic_pipeline.pkl")

st.title("Titanic Survival Predictor")

st.markdown("Enter passenger details to predict survival:")


pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("No. of Siblings/Spouses aboard", 0, 8, 0)
parch = st.slider("No. of Parents/Children aboard", 0, 6, 0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
deck = st.selectbox("Deck", ['U', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])  

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Deck": deck
}])


if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader(f"Prediction: {'🟢 Survived' if prediction == 1 else '🔴 Did Not Survive'}")
    st.write(f"Probability of Survival: {prob[1]*100:.2f}%")

    # Probability visualization
    fig, ax = plt.subplots()
    sns.barplot(x=["Did Not Survive", "Survived"], y=prob, palette='coolwarm', ax=ax)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
