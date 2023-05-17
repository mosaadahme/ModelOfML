from flask import Flask , request , jsonify , render_template
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# Create flask app
app = Flask(__name__,template_folder='Templates')

# Load pickle model
model = pickle.load(open("model.pkl" , "rb"))
scaler = pickle.load(open('scaler.pkl',"rb"))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/form')
def form():
    return render_template("form.html")


@app.route('/result' , methods=["POST"])
def result(): 
    nitrogen = int(request.form['nitrogen'])
    p = int(request.form['potassium'])
    k = int(request.form['kpo'])
    temp = float(request.form['temprature'])
    hd = float(request.form['hd'])
    ph = float(request.form['ph'])
    rain = float(request.form['rain'])

    data = [nitrogen , p, k , temp , hd, ph , rain]
    vect = [np.array(data)]
    vect = scaler.transform(vect)
    model_prediction = model.predict(vect)
    return render_template('result.html',label = model_prediction)

if __name__ == '__main__':
	app.run(debug=True)