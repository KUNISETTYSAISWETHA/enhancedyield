from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
filepath='C:/Users/kunis/Documents/UX design/dtrr.pkl'
file='C:/Users/kunis/Documents/UX design/preprocesserr.pkl'
dtrr=pickle.load(open(filepath,'rb'))
preprocesserr=pickle.load(open(file,'rb'))
app=Flask(__name__) #initializes a flask web application
@app.route('/') # Defines a route for the root URL. When users access the root URL
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Crop_Year=request.form['Crop_Year']
        Area = request.form['Area']
        Production = request.form['Production']
        Annual_Rainfall=request.form['Annual_Rainfall'] 
        Fertilizer = request.form['Fertilizer']
        Pesticide=request.form['Pesticide']
        Crop = request.form['Crop']
        State = request.form['State']
        try:
        
            Production  = float(Production)
        except ValueError:
            # Handle the case where investment_str is not a valid float
            return render_template('index.html', error_message="Invalid investment value")

        features = np.array([[Crop_Year,Area,Production,Annual_Rainfall,Fertilizer,Pesticide,Crop,State]])
        transformed_features = preprocesserr.transform(features)
        predicted_value = dtrr.predict(transformed_features).reshape(1, -1)
        s=Production-predicted_value
        if(s>0):
            result='profit'
        else:
            result='loss'
        return render_template('index.html',predicted_value=predicted_value,result=result)





if __name__=='__main__':
    app.run(debug=True)