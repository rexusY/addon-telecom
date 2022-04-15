from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
#from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import talos as ta

#import tensorflow as tf
#import tensorflow.keras as keras


# pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

app = Flask(__name__, template_folder='template')
logreg = joblib.load(open('logreg.joblib', 'rb'))
#logreg = joblib.load(open('logreg_nonsmote.joblib', 'rb'))
#logreg = joblib.load(open('pipe_log.joblib', 'rb'))
#pipeline = joblib.load(open('pipeline_transformer.joblib', 'rb'))
@app.route('/')
def home():
    return render_template("homepage.html")

def get_data():
    Billing = request.form.get('Billing')
    AdditionalBilling = request.form.get('AdditionalBilling')
    UseInet	 = request.form.get('Use Inet')
    UseTV = request.form.get('Use TV')
    Type = request.form.get('Type')
    InitialPackage = request.form.get('InitialPackage')
    OfferedPackage = request.form.get('OfferedPackage')
    AdditionalService = request.form.get('AdditionalService')
    OfferingTime = request.form.get('OfferingTime')
    LoS = request.form.get('LoS')
    Area = request.form.get('Area')
    


    d_dict = {'Billing': [Billing] , 
              'AdditionalBilling': [AdditionalBilling], 
              'Use Inet': [UseInet],
              'Use TV': [UseTV],
              'LoS': [LoS],
			  #'Type': [0],
			  #'InitialPackage': [0],
			  #'OfferedPackage': [0],
			  #'AdditionalService': [0],
              #'OfferingTime': [0],  
              #'Area': [0] ,
              'Type': [Type],
			  'InitialPackage': [InitialPackage],
			  'OfferedPackage': [OfferedPackage],
			  'AdditionalService': [AdditionalService],
              'OfferingTime': [OfferingTime],  
              'Area': [Area] }




    replace_list = [Billing, AdditionalBilling, UseInet,UseTV, Type,
                    InitialPackage, OfferedPackage, AdditionalService,
                    OfferingTime, LoS, Area]

    for key, value in d_dict.items():
        if key in replace_list:
            d_dict[key] = 1


    return pd.DataFrame.from_dict(d_dict, orient='columns')


@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    
    #entry_transformed = pipeline.transform(df)
    prediction = logreg.predict(df)
    outcome = 'Accept'
    if prediction == 0:
        outcome = 'Decline'

    return render_template('results.html', tables = [df.to_html(classes='data', header=True)],
                           result = outcome)



if __name__=="__main__":
    app.run(debug=True)