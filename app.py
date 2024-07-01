from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        company = request.form['company']
        prevclose = request.form['PrevClose']
        openvalue = request.form['Open']
        high = request.form['High']
        low = request.form['Low']
        last = request.form['Last']
        vwap = request.form['VWAP']

        
        if(company == 'adani'):
            model = pickle.load(open("./scripts/classifier_adaniports.pkl", "rb"))
        elif(company == 'hdfc'):
            model = pickle.load(open("./scripts/classifier_hdfc.pkl", "rb"))
        elif(company == 'icici'):
            model = pickle.load(open("./scripts/classifier_icici.pkl", "rb"))
        elif(company == 'tatamotors'):
            model = pickle.load(open("./scripts/classifier_tatamotors.pkl", "rb"))
        elif(company == 'tatasteel'):
            model = pickle.load(open("./scripts/classifier_tatasteel.pkl", "rb"))
        elif(company == 'tcs'):
            model = pickle.load(open("./scripts/classifier_tcs.pkl", "rb"))
        else:
            return 'Error'
        
        data_yes = [[prevclose,openvalue,high,low,last,vwap]]
        test_yes = pd.DataFrame(data_yes,columns=['Prev Close','Open','High','Low','Last','VWAP'])
        result = model.predict(test_yes)
        result2f = round(result[0],2)
        return render_template("result.html", res=result2f)
    else:
        return 'Error'

if __name__ == "__main__":
    app.run(debug=True)