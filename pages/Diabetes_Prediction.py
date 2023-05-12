import streamlit as st 
import joblib
import pandas as pd

st.write("# Diabetes Prediction")

col1, col2, col3 = st.columns(3)

#getting user input

gender = col1.selectbox("Enter your gender",["Male", "Female"])

age = col2.number_input("Enter your age")

hypertension = col3.selectbox("Do you have hypertension?",["Yes","No"])

heart_disease = col1.selectbox("Do you have heart disease?",["Yes","No"])

smoking_history = col2.selectbox("Are you a smoker?",["never", "ever", "former", "not current", "current", "No Info"])

bmi = col3.number_input("Enter your BMI")

HbA1c_level = col2.number_input("Enter your HbA1c level")

blood_glucose_level = col3.number_input("Enter your blood glucose level")

df_pred = pd.DataFrame([[gender,age,hypertension,heart_disease,smoking_history, bmi, HbA1c_level,blood_glucose_level]],

columns= ['gender','age','hypertension','heart_disease','smoking_history', 'bmi', 'HbA1c_level','blood_glucose_level'])

df_pred['gender'] = df_pred['gender'].apply(lambda x: 1 if x == 'Male' else 0)

df_pred['hypertension'] = df_pred['hypertension'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['heart_disease'] = df_pred['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)

def transform(data):
    result = 5
    if(data=='current'):
        result = 0
    elif(data=='not current'):
        result = 1
    elif(data=='former'):
        result = 2
    elif(data=='ever'):
        result = 3
    elif(data=='never'):
        result = 4
    return(result)

df_pred['smoking_history'] = df_pred['smoking_history'].apply(transform)

model = joblib.load('diabetes_rfc_model.pkl')
prediction = model.predict(df_pred)

if st.button('Predict'):
    
    if(prediction[0]==0):
        st.write('<p class="big-font">You likely do not have diabetes.</p>',unsafe_allow_html=True)

    else:
        st.write('<p class="big-font">You likely do have diabetes.</p>',unsafe_allow_html=True)
        
st.markdown("This is my own version of the Heart Disease Prediction application. I found a diabetes dataset on Kaggle built from Electronic Health Records (EHRs) collected from multiple health care providers. I used a random forest classifier that has 97% accuracy from the provided data.")