# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def load_page():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    inputs = [request.form.get(f'query{i}', '') for i in range(1, 20)]

    new_df = pd.DataFrame([inputs], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'tenure'
    ])

    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), bins=range(1, 80, 12), right=False, labels=labels)
    
    df_2.drop(columns=['tenure'], inplace=True)
    
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                          'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])
    
    prediction = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:,1]

    if prediction == 1:
        outcome = "This customer is likely to be churned!!"
    else:
        outcome = "This customer is likely to continue!!"
    
    confidence = f"Confidence: {probability[0] * 100:.2f}%"

    return render_template('home.html', output1=outcome, output2=confidence,
                           **{f'query{i}': request.form.get(f'query{i}', '') for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)
