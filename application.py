from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Import ridge regressor and standard scaler
ridge_model = pickle.load(open("models/ridge.pkl", 'rb'))
standard_scaler = pickle.load(open("models/scaler.pkl", 'rb'))

@app.route("/") 
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=["GET","POST"])
def predictdata():
    if request.method=="POST":
        MedInc	=float(request.form.get('MedInc'))
        HouseAge=float(request.form.get('HouseAge'))	
        AveRooms	=float(request.form.get('AveRooms'))
        AveBedrms=float(request.form.get('AveBedrms'))
        Population	=float(request.form.get('Population'))
        AveOccup	=float(request.form.get('AveOccup'))
        Latitude	=float(request.form.get('Latitude'))
        Longitude=float(request.form.get('Longitude'))
        temp_new_data=standard_scaler.transform([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
        result=ridge_model.predict(temp_new_data)
        return render_template('home.html',results=result[0])
        
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0" )