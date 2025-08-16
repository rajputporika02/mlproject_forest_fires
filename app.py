from flask import Flask ,request,jsonify,render_template 
import pickle 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 



app=Flask(__name__)

##import models
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
scaler_model=pickle.load(open('models/scaler.pkl','rb'))



@app.route("/")#######################
def index():############################      INDEX PAGE 
    return render_template('index.html')#########




@app.route('/predictdata',methods=['GET','POST']) #get-google search , #post-searching after

def predict_datapoint():
    if request.method=='POST': #this will run when we enter the data  (post)                            
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html') #when we do not enter the data (get)


if __name__=="__main__":
    app.run(host='0.0.0.0')
