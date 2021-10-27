
import streamlit as st

import pandas as pd
import numpy as np
from joblib import load
# write title
st.title('Churn prediction app')

#creating a file upload option

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:

    x_df = pd.read_csv(uploaded_file)
    categorical_features=['gender','PaymentMethod','OnlineSecurity','TechSupport','Contract']
    for col in categorical_features:
        x_df[col] = x_df[col].astype("category").cat.codes

else:
    def input_features():
        TotalCharges = st.sidebar.slider('Total Charges', 50.0, 10000.0, 50.0)
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 15.0, 120.0, 15.0)
        tenure = st.sidebar.slider('tenure', 0.0, 72.0, 0.0)
        Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'two year'))
        PaymentMethod = st.sidebar.selectbox('PaymentMethod', ('Bank transfer (automatic)',
                                                          'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        OnlineSecurity = st.sidebar.selectbox('OnlineSecurity', ('No', "Yes", "No internet service "))
        TechSupport = st.sidebar.selectbox('TechSupport', ('No', "No internet service ", "Yes"))
        gender = st.sidebar.selectbox('gender', ('Male', 'Female'))




        data = {'TotalCharges': [TotalCharges],
                'MonthlyCharges': [MonthlyCharges],
                'tenure': [tenure],
                'Contract': [Contract],
                'PaymentMethod': [PaymentMethod],
                'OnlineSecurity': [OnlineSecurity],
                'TechSupport': [TechSupport],
                'gender': [gender]
                }
        features = pd.DataFrame(data)

        return features
    x_df = input_features()
    categorical_features = ['gender', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'Contract']
    for col in categorical_features:
        x_df[col] = x_df[col].astype("category").cat.codes

model = load(filename='mychurn_model.joblib')
if st.button('Predict churn'):
    prediction = model.predict(x_df)
    prediction_probability = model.predict_proba(x_df)
    churn = np.array(['No','Yes'])
    st.subheader('Predicted churn')
    st.write(churn[model.predict(x_df)])

    st.subheader('Prediction Probability')

    st.write(prediction_probability)
    st.success('The Probability of customer churn is {}'.format(prediction))

