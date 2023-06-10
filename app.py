from flask import Flask ,render_template,jsonify,url_for,flash,redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField,BooleanField
from wtforms.validators import DataRequired,length,Email,Regexp,EqualTo
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import json
import requests
from email_validator import validate_email
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import matplotlib.pyplot as plt



app=Flask(__name__)
app.config['SECRET_KEY']='293b8cad24d6f3600ee5a2d67dc1c3efc48f19d0f84b5c2c229e997409a06d20'



@app.route("/index")
def index():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {'X-CMC_PRO_API_KEY': "4116c941-a133-4abf-b6c4-bd6cb321b499"}
    parameters_btc = {'symbol': 'BTC'}
    response_btc = requests.get(url, headers=headers, params=parameters_btc)
    data_btc = response_btc.json()
    price_btc = data_btc['data']['BTC']['quote']['USD']['price']
    market_cap_btc = data_btc['data']['BTC']['quote']['USD']['market_cap']

    parameters_eth = {'symbol': 'ETH'}
    response_eth = requests.get(url, headers=headers, params=parameters_eth)
    data_eth = response_eth.json()
    price_eth = data_eth['data']['ETH']['quote']['USD']['price']
    market_cap_eth = data_eth['data']['ETH']['quote']['USD']['market_cap']

    parameters_ada = {'symbol': 'ADA'}
    response_ada = requests.get(url, headers=headers, params=parameters_ada)
    data_ada = response_ada.json()
    price_ada = data_ada['data']['ADA']['quote']['USD']['price']
    market_cap_ada = data_ada['data']['ADA']['quote']['USD']['market_cap']
    return render_template('public/index.html',price_btc=price_btc,price_eth=price_eth,price_ada=price_ada)
   


@app.route("/about")
def about():
    return render_template("public/about.html")




class inscrireform(FlaskForm):
    fname = StringField('First Name',validators=[DataRequired(),length(min=2,max=25)])
    lname = StringField('Last Name',validators=[DataRequired(),length(min=2,max=25)])
    username = StringField('Username',validators=[DataRequired(),length(min=2,max=25)])
    email = StringField("Email",validators=[DataRequired(),Email()])
    password = PasswordField('Password',validators=[DataRequired(),Regexp("^(?=.*[A-Z])(?=.*[a-z])(?=.*[@$!%*?&_])[A-Za-z\d@$!%*?&_]{8,32}$")])
    confirm_password = PasswordField('Confirm_password',validators=[DataRequired(),EqualTo('password')])
    submit = SubmitField("Sign Up")

class loginform(FlaskForm):
    
    email = StringField("Email",validators=[DataRequired(),Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    remember = BooleanField('Remember Me') 
    submit = SubmitField("Login")


@app.route("/login",methods=["GET","POST"])
def login():
    form= loginform()
    if form.validate_on_submit():
        if form.email.data =='ouyoussmeryem@gmail.com' and form.password.data == "PASS??word123":
            flash("You have been logged in !!","success")
            return redirect("/index")
        else :
            flash("Login Unsuccessful , please check credentials","danger")
    return render_template("public/login.html",form=form)



@app.route("/inscrire",methods=["GET","POST"])
def inscrire():
    form= inscrireform()
    if form.validate_on_submit():
        flash(f"Account created successfully for {form.username.data}","success")
        return redirect("/index")
    return render_template("public/inscription.html",form=form)

@app.route("/predir/eth")
def predir_eth():
    # Charger et nettoyer les données
    df = pd.read_csv("ETH-USD.csv")
    df = df.dropna()

    # Afficher les données visuellement
    df.plot(x="Date", y="Close")
    plt.xticks(rotation=45)

    # Créer le modèle
    model = LinearRegression()

    # Entraîner le modèle
    X = df[['Open', 'High', 'Low', 'Volume']]
    X = X[:int(len(df)-1)]
    y = df['Close']
    y = y[:int(len(df)-1)]
    model.fit(X, y)  # Entraînement du modèle

    # Faire les prédictions
    new_data = df[['Open', 'High', 'Low', 'Volume']].tail(1)
    prediction = model.predict(new_data)
    actual_price = df[['Close']].tail(1).values[0][0]

    # Obtenir le prix actuel du Bitcoin via une API
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {'X-CMC_PRO_API_KEY': "4116c941-a133-4abf-b6c4-bd6cb321b499"}
    parameters_btc = {'symbol': 'ETH'}
    response_btc = requests.get(url, headers=headers, params=parameters_btc)
    data_btc = response_btc.json()
    price_btc = data_btc['data']['ETH']['quote']['USD']['price']
    market_cap_btc = data_btc['data']['ETH']['quote']['USD']['market_cap']

    difference = price_btc - prediction
    current_date = datetime.datetime.now()  
    
    return render_template("public/ethereum.html", prediction=prediction, actual_price=actual_price, price_btc=price_btc, difference=difference, current_date=current_date)


@app.route("/prediction",methods=['POST'])
def prediction():
    
    
    return render_template("public/prediction.html")

@app.route("/bitcoinRF")
def bitcoinRF():
    #collect and clean the data
    df = pd.read_csv("btc.csv")
    df = df.dropna()

    #show the data visually
    df.plot(x="Date", y="Close")
    plt.xticks(rotation=45)

    #Create the model
    model = RandomForestRegressor()

    #Train the model
    X=df[['Open','High', 'Low' , 'Volume']]
    X=X[:int(len(df)-1)]
    y=df['Close']
    y=y[:int(len(df)-1)]
    model.fit(X,y) #training model

    RandomForestRegressor()

    #Test the model
    predictions = model.predict(X)
    #Make the predictions
    new_data=df[['Open','High', 'Low' , 'Volume']].tail(1)
    prediction = model.predict(new_data)
    

    #Make the predictions
    new_data=df[['Open','High', 'Low' , 'Volume']].tail(1)
    prediction = model.predict(new_data)
    ActuelPrice=df[['Close']].tail(1).values[0][0]

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {'X-CMC_PRO_API_KEY': "4116c941-a133-4abf-b6c4-bd6cb321b499"}
    parameters_btc = {'symbol': 'BTC'}
    response_btc = requests.get(url, headers=headers, params=parameters_btc)
    data_btc = response_btc.json()
    price_btc = data_btc['data']['BTC']['quote']['USD']['price']
    market_cap_btc = data_btc['data']['BTC']['quote']['USD']['market_cap']

    difference = price_btc - prediction 
    current_date = datetime.datetime.now()  
    
    return render_template("public/bitcoinRF.html",prediction=prediction,ActuelPrice=ActuelPrice,price_btc=price_btc,difference=difference,current_date=current_date)


@app.route("/ETHRF")
def ETHRF():
    #collect and clean the data
    df = pd.read_csv("ETH-USD.csv")
    df = df.dropna()

    #show the data visually
    df.plot(x="Date", y="Close")
    plt.xticks(rotation=45)

    #Create the model
    model = RandomForestRegressor()

    #Train the model
    X=df[['Open','High', 'Low' , 'Volume']]
    X=X[:int(len(df)-1)]
    y=df['Close']
    y=y[:int(len(df)-1)]
    model.fit(X,y) #training model

    RandomForestRegressor()

    #Test the model
    predictions = model.predict(X)
    #Make the predictions
    new_data=df[['Open','High', 'Low' , 'Volume']].tail(1)
    prediction = model.predict(new_data)
    

    #Make the predictions
    new_data=df[['Open','High', 'Low' , 'Volume']].tail(1)
    prediction = model.predict(new_data)
    ActuelPrice=df[['Close']].tail(1).values[0][0]

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {'X-CMC_PRO_API_KEY': "4116c941-a133-4abf-b6c4-bd6cb321b499"}
    parameters_btc = {'symbol': 'ETH'}
    response_btc = requests.get(url, headers=headers, params=parameters_btc)
    data_btc = response_btc.json()
    price_btc = data_btc['data']['ETH']['quote']['USD']['price']
    market_cap_btc = data_btc['data']['ETH']['quote']['USD']['market_cap']

    difference = price_btc - prediction 
    current_date = datetime.datetime.now()  
    
    return render_template("public/ETHRF.html",prediction=prediction,ActuelPrice=ActuelPrice,price_btc=price_btc,difference=difference,current_date=current_date)






@app.route("/bitcoinREGLIN")
def bitcoinREGLIN():
    # Charger et nettoyer les données
    df = pd.read_csv("btc.csv")
    df = df.dropna()

    # Afficher les données visuellement
    df.plot(x="Date", y="Close")
    plt.xticks(rotation=45)

    # Créer le modèle
    model = LinearRegression()

    # Entraîner le modèle
    X = df[['Open', 'High', 'Low', 'Volume']]
    X = X[:int(len(df)-1)]
    y = df['Close']
    y = y[:int(len(df)-1)]
    model.fit(X, y)  # Entraînement du modèle

    # Faire les prédictions
    new_data = df[['Open', 'High', 'Low', 'Volume']].tail(1)
    prediction = model.predict(new_data)
    actual_price = df[['Close']].tail(1).values[0][0]

    # Obtenir le prix actuel du Bitcoin via une API
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {'X-CMC_PRO_API_KEY': "4116c941-a133-4abf-b6c4-bd6cb321b499"}
    parameters_btc = {'symbol': 'BTC'}
    response_btc = requests.get(url, headers=headers, params=parameters_btc)
    data_btc = response_btc.json()
    price_btc = data_btc['data']['BTC']['quote']['USD']['price']
    market_cap_btc = data_btc['data']['BTC']['quote']['USD']['market_cap']

    difference = price_btc - prediction
    current_date = datetime.datetime.now()  
    
    return render_template("public/bitcoinREGLIN.html", prediction=prediction, actual_price=actual_price, price_btc=price_btc, difference=difference, current_date=current_date)







@app.route("/bitcoinREGLINcinqJ")
def bitcoinREGLINcinqJ():
    # Charger et nettoyer les données
    df = pd.read_csv("btc.csv")
    df = df.dropna()

    # Afficher les données visuellement
    df.plot(x="Date", y="Close")
    plt.xticks(rotation=45)

    # Créer le modèle
    model = LinearRegression()

    # Entraîner le modèle
    X = df[['Open', 'High', 'Low', 'Volume']]
    X = X[:int(len(df)-1)]
    y = df['Close']
    y = y[:int(len(df)-1)]
    model.fit(X, y)  # Entraînement du modèle

    # Faire les prédictions
    new_data = df[['Open', 'High', 'Low', 'Volume']].tail(1)
    prediction = model.predict(new_data)
    actual_price = df[['Close']].tail(1).values[0][0]

    # Faire les prédictions pour les cinq prochains jours
    next_five_days = pd.date_range(start=df['Date'].iloc[-1], periods=6)[1:]
    next_five_days_data = np.repeat(new_data.values, 5, axis=0)
    next_five_days_predictions = model.predict(next_five_days_data)

    # Créer un DataFrame pour stocker les dates et les prédictions
    predictions_df = pd.DataFrame({'Date': next_five_days, 'Prediction': next_five_days_predictions})
    
    # Créer les données de trace pour le graphique matplotlib
    dates = predictions_df['Date']
    predictions = predictions_df['Prediction']

    # Afficher le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(dates, predictions, marker='o')
    plt.title('Bitcoin Price Predictions for the Next 5 Days')
    plt.xlabel('Date')
    plt.ylabel('Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Sauvegarder le graphique dans un fichier
    plt.savefig('image.png')
    
    return render_template("public/bitcoinREGLINcinqJ.html",predictions=predictions_df)

if __name__ == "__main__":
    app.run()










