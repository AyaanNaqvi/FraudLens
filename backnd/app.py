from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report
app = Flask(__name__)
CORS(app)
logisticreg=joblib.load('logisticreg.pkl')
randomfor=joblib.load('randomfor.pkl')
XG=joblib.load('XGboost.pkl')

@app.route('/compare', methods=['GET'])
def compare():
    df=pd.read_csv('creditcard.csv')
    df = cleandata(df)
    X=df.drop('Class',axis=1)
    Y=df['Class']
    results=run_models(X,Y)
    return jsonify(results)


def cleandata(df):

    df['Amount']=StandardScaler().fit_transform(df[['Amount']])   
    return df

def run_models(X,Y):
    logisticreg_prediction=logisticreg.predict(X)
    randomfor_prediction=randomfor.predict(X)
    XG_prediction=XG.predict(X)
    logisticreg_accuracy=accuracy_score(Y, logisticreg_prediction)
    randomfor_accuracy=accuracy_score(Y,randomfor_prediction)
    XG_accuracy=accuracy_score(Y,XG_prediction)
    logisticreg_classification=classification_report(Y,logisticreg_prediction, output_dict=True)
    randomfor_classification=classification_report(Y,randomfor_prediction, output_dict=True)
    XG_classification=classification_report(Y,XG_prediction, output_dict=True)
    return {
    'logistic': {'accuracy': logisticreg_accuracy, 'report': logisticreg_classification},
    'random_forest': {'accuracy': randomfor_accuracy, 'report': randomfor_classification},
    'xgboost': {'accuracy': XG_accuracy, 'report': XG_classification}
}

@app.route('/upload/<model_name>',methods=['POST'])
def uplaod(model_name):
    file=request.files.get('file')
    df=pd.read_csv(file)
    df=cleandata(df)
    X=df.drop('Class',axis=1,errors='ignore')
    models = {
    'logistic': logisticreg,
    'random_forest': randomfor,
    'xgboost': XG
}
    model = models[model_name]
    prediction=model.predict(X)
    fraud_count = int(prediction.sum())
    not_fraud = len(prediction) - fraud_count
    return jsonify({'fraud': fraud_count, 'not_fraud': not_fraud})







if __name__ == '__main__':
    app.run(debug=True)
