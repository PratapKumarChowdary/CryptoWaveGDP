from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
#from pydantic import BaseModel
from pymongo import MongoClient
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import yfinance as yf
#from forex_python.converter import CurrencyRates
import requests
import os
import pandas as pd
import copy
#import itertools
#import warnings

import numpy as np
from pandas_datareader import data as pdr
#import matplotlib.pyplot as plt

# from prophet.plot import plot_plotly, plot_components_plotly
#from prophet import Prophet

#import plotly.express as px

#from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


load_dotenv()

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with your MongoDB URI
mongo_uri = os.getenv("DATABASE_URL")
mongo_client = MongoClient(mongo_uri)

try:
    mongo_client.admin.command('ping')
    print("You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = mongo_client["crypto"]  # Replace "mydb" with your database name
users_collection = db["users"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



# Secret key for JWT token
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 172800000


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def getCurrencyConversionData():
    try:
        apiKey = os.getenv("EXCHANGE_API_KEY")
        
        if not apiKey:
            raise ValueError("EXCHANGE_API_KEY environment variable is not set.")
        
        currencyConversionUrl = f"http://data.fixer.io/api/latest?access_key={apiKey}&symbols=INR,USD,EUR"
        
        response = requests.get(currencyConversionUrl)
        
        if response.status_code == 200:
            currencyData = response.json()
            
            euroToInr = currencyData["rates"]["INR"]
            euroToUsd = currencyData["rates"]["USD"]
            
            usdToEuro = 1 / euroToUsd
            usdToInr = euroToInr / euroToUsd
            return usdToEuro,usdToInr
        else:
            raise Exception(f"API request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

### Helper Functions
# Function to scrape crypto data from Yahoo Finance
def scrape_crypto_data(crypto_name, time_format, frequency='daily'):
    yf.pdr_override()
    
    if frequency == 'daily':
        interval = '1d'
    elif frequency == 'monthly':
        interval = '1mo'
    elif frequency == 'yearly':
        interval = '5y'
    else:
        raise ValueError("Invalid frequency. Choose from 'daily', 'monthly', or 'yearly'")
    
    start_date = '2019-01-01'
    # end_date = '2023-07-31'
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(end_date)
    df = pdr.get_data_yahoo(crypto_name, start=start_date, end=end_date, interval=interval)
    
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df['ds'] = df['ds'].dt.strftime(time_format)
    
    return df

# Function to remove outliers using Local Outlier Factor
def remove_outliers(df):
    X = df[['y']]
    lof = LocalOutlierFactor(contamination=0.05)
    outliers = lof.fit_predict(X)
    
    df['outlier'] = outliers
    clean_df = df[df['outlier'] != -1]
    
    return clean_df.drop(columns=['outlier'])

# Function to handle missing values by replacing with mean
def handle_missing_values(df):
    df['y'].fillna(df['y'].mean(), inplace=True)
    return df


# Function to split data into training and testing sets
def train_test_split(df, ratio=0.8):
    split_index = int(ratio * len(df))
    train_data = df[:split_index]
    test_data = df[split_index:]
    return train_data, test_data

# Function to evaluate the model using MAPE, MSE, and RMSE
def evaluate_forecast(test_df, forecast_df):
    y_true = test_df['y'].values
    y_pred = forecast_df['yhat'].values[-len(y_true):]
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print("Mape" + mape)
    print(rmse)
    
    return mape, rmse


""" Multivariate LSTM """
def create_XY_Sequences(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)



def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50, input_shape=(window_size, features), return_sequences=True))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

from dateutil.relativedelta import relativedelta
def generate_forecast_df(end_date, freq, forecast):
    # Convert end_date to datetime
    end_date = datetime.strptime(end_date, "%Y-%m-%d")


    # Define the frequency
    if freq == 'monthly':
        next_month = end_date + relativedelta(months=1)
        forecast_dates = [next_month + relativedelta(months=i) for i in range(forecast)]

    elif freq == 'daily':
        forecast_dates = [end_date + relativedelta(days=i) for i in range(1, forecast + 1)]
    
    else:
        raise ValueError("Invalid frequency. Choose 'monthly' or 'daily'.")

    # Create a DataFrame
    forecast_df = pd.DataFrame(forecast_dates, columns=['Forecast Dates'])

    return forecast_df

# Function to train and evaluate MutliVar LSTM model
def LSTM_forecasting(crypto_name, time_format, frequency, forecast_days):
    
    global window_size, features

    freq = "D" if frequency == "daily" else "M" if frequency == "monthly" else "Y"
    
    # Scrape data
    crypto_data = scrape_crypto_data(crypto_name, time_format, frequency)
    print(crypto_data)
    
    clean_data = crypto_data[['Open', 'High', 'Low', 'y']]
    features = len(clean_data.columns)
    # print(features)
    
    # Train-test split
    train_data, test_data = train_test_split(clean_data, ratio=0.9)
    # print(train_data.shape)
    # print(test_data.shape)
    
    # Scale Data
    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    window_size = 4
    trainX, trainY = create_XY_Sequences(train_data_scaled, window_size)
    testX, testY = create_XY_Sequences(test_data_scaled, window_size)
        
    # Split the data into training and validation sets
    validation_split = 0.2  # Adjust the percentage as needed
    split_index = int(len(trainX) * (1 - validation_split))
    trainX, X_val = trainX[:split_index], trainX[split_index:]
    trainY, y_val = trainY[:split_index], trainY[split_index:]

    print(trainX.shape)
    print(testX.shape)
    
    # Build the LSTM model and Train using GridSearch
    grid_model = KerasRegressor(build_fn = build_model, verbose=1, validation_data = (testX,testY))
    
    parameters = {'batch_size' : [16, 32, 64],
                'epochs' : [10, 50, 100],
                'optimizer' : ['adam','Adadelta'] }
    
    grid_search  = GridSearchCV(estimator = grid_model,
                                param_grid = parameters,
                                cv = 2)
    
    # grid_search = grid_search.fit(trainX, trainY)
    # model = grid_search.best_estimator_.model

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, features))) # , return_sequences=True
    model.add(Dropout(0.8))
    # model.add(LSTM(64))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(trainX, trainY, epochs=5, batch_size=4, validation_data=(X_val, y_val))

    
    # Make predictions
    prediction = model.predict(testX)
    
    # Change Shape & Inverse Transform Predictions
    pred_C = np.repeat(prediction, features, axis=-1)
    pred = scaler.inverse_transform(np.reshape(pred_C, (len(prediction), features)))[:,0]
    
    # Change Shape & Inverse Transform Test Set Y
    testY_C = np.repeat(testY, features, axis=-1)
    orig = scaler.inverse_transform(np.reshape(testY_C, (len(testY), features)))[:,0]
    
    # print("Pred Values-- " ,pred)
    # print("\nOriginal Values-- " ,original)

    # Evaluate
    mape = mean_absolute_percentage_error(orig, pred)
    mse = mean_squared_error(orig, pred)
    rmse = np.sqrt(mse)
    print("Mape")
    print(mape)
    print("RMSE")
    print(rmse)
    
    scoreDict['LSTM'] = mape, rmse
    

    # plt.plot(orig, color = 'red', label = 'Real Stock Price')
    # plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
    # plt.title('Stock Price Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Google Stock Price')
    # plt.legend()
    # plt.show()
    

    ### Exogenuous Features Out of Sample Forecast using HoltWinters
    exogDF = clean_data.drop('y', axis=1) # Get predictor dataframe
    exogCols = exogDF.columns # Get predictor Columns
    print(exogCols)
    predictorDict = {} # Dict to store future forecast for each predictor

    for i in exogCols: # Loop over each predictor column
        print(f'Forecasting Predictor: {i}')
        temp = exogDF[i]
        
        modelHW = ExponentialSmoothing(temp, trend="add", seasonal="add", seasonal_periods=7) # Train HoltWinter model on predictor historical data
        fit = modelHW.fit()

        forecast_HW = fit.forecast(steps=forecast_days) # Forecast predictor value
        forecast_HW = forecast_HW.round().astype(int)
        predictorDict[i] = forecast_HW

    osNullDF = pd.DataFrame(predictorDict) # Combine predictors forecast in a dataframe
    osNullDF = osNullDF.rename_axis('date')
    print(osNullDF)


    # Take last len(window_size) values from Dataset
    df_Window_Past = clean_data.iloc[-window_size:,:]

    # Set future "y" to nan
    osNullDF["y"] = 0

    # Get Out Of Sample Exog Features
    # osNullDF = osNullDF[["Open","High","Low","y"]]
    
    # Scale OoS Exog and Y Column
    past_Scaled = scaler.transform(df_Window_Past)
    OS_Scaled = scaler.transform(osNullDF)

    # Create Dataframe and Y col to Nan
    OS_Scaled_DF = pd.DataFrame(OS_Scaled)
    OS_Scaled_DF.iloc[:,0] = np.nan

    # Combine last Window_size data with OoS data
    FinalDF=pd.concat([pd.DataFrame(past_Scaled),OS_Scaled_DF]).reset_index().drop(["index"],axis=1)
    print(FinalDF)


    # Loop over each Future entry and make a prediction using Exog Features and trained LSTM Model
    FinalDF_Scaled=FinalDF.values
    compData = []
    window_size
    for i in range(window_size, len(FinalDF_Scaled)):
        exog_Data = []
        exog_Data.append(
        FinalDF_Scaled[i-window_size :i, 0:FinalDF_Scaled.shape[1]])
        exog_Data = np.array(exog_Data)
        OS_Pred = model.predict(exog_Data)
        compData.append(OS_Pred)
        FinalDF.iloc[i,0] = OS_Pred

    # Inverse Transform Out of Sample Prediction
    OS_Pred_List=np.array(compData)
    OS_Pred_List=OS_Pred_List.reshape(-1,1)
    OS_PredC = np.repeat(OS_Pred_List,features, axis=-1)
    OS_Pred_Final = scaler.inverse_transform(np.reshape(OS_PredC,(len(OS_Pred_List),features)))[:,0]
    print("LSTM Forecast")
    print(OS_Pred_Final)


    # TestPred DF
    train_data, test_data = train_test_split(crypto_data, ratio=0.9)
    lstmDF = test_data[['ds','y']]
    lstmDF = lstmDF.iloc[window_size:,:]
    lstmDF['LSTM_Pred'] = pred.tolist()
    lstmDF = lstmDF.reset_index(drop=True)


    # Out of Sample DF
    end_date = crypto_data['ds'].max()
    print(crypto_data['ds'].max())
    forecast_df = generate_forecast_df(end_date, frequency, forecast_days)
    forecast_df['LSTM_Frc'] = OS_Pred_Final
    forecast_df = forecast_df.rename(columns={'Forecast Dates': 'ds'})
    print(forecast_df)

    resultDF = crypto_data[['ds','y']].copy()
    print(resultDF)
    resultDF = resultDF.append(forecast_df)
    resultDF = resultDF.reset_index(drop=True)
    resultDF['ds'] = pd.to_datetime(resultDF['ds']).dt.strftime('%Y-%m-%d')
    resultDF.to_csv('forecast.csv', index=False)
    print(resultDF)

    # return OS_Pred_Final, mape, rmse, lstmDF, forecast_df
    return resultDF

@app.post("/register")
async def register_user(user:dict):
    email = user["email"]
    password = user["password"]
    firstName = user["firstName"]
    lastName = user["lastName"]
    
    existing_user = users_collection.find_one({"email": email})

    if existing_user:
        raise HTTPException(status_code=409, detail="email already exists")

    hashed_password = pwd_context.hash(password)

    user_data = {
        "email": email,
        "password": hashed_password,
        "firstName": firstName,
        "lastName": lastName
    }

    mongo_client["crypto"]["users"].insert_one(user_data)

    return {"message": "User registered successfully"}


@app.post("/login")
async def login_user(user:dict):
    email = user["email"]
    password = user["password"]

    existing_user = users_collection.find_one({"email": email})

    if existing_user is None or not pwd_context.verify(password, existing_user["password"]):
        raise HTTPException(
            status_code=400, detail="Incorrect email or password")

    data = {"sub": str(existing_user["email"]), "exp": datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

    user = {"email": existing_user["email"], "token": access_token}

    return user


@app.get("/userProfile")
async def protected_route(authorization: str = Header(...)):

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="you are not authorized to perform this operation")

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        existing_user = users_collection.find_one({"email": payload["sub"]})

        return {"email": existing_user["email"]}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/markets")
async def fetchMarkets(authorization: str = Header(...)):

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="you are not authorized to perform this operation")

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        Symbols = ["BTC-USD", "ETH-USD", "XRP-USD", "TRX-USD", "MATIC-USD", "ADA-USD",
                   "SHIB-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "USDC-USD", "USDT-USD"]

        tickers = yf.Tickers(Symbols)

        historical_data = tickers.history(period="1d")


        formattedData = []
        for symbol in Symbols:
            priceDict = {}
            priceDict["symbol"] = symbol
            priceDict["low"] = historical_data["Low"][symbol][0]
            priceDict["open"] = historical_data["Open"][symbol][0]
            priceDict["high"] = historical_data["High"][symbol][0]
            priceDict["name"] = symbol.split("-")[0]

            formattedData.append(priceDict)

        return {"markets": formattedData}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/markets/compare")
async def fetchMarkets(data:dict,authorization: str = Header(...)):

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="you are not authorized to perform this operation")

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        #Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        #Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        intervalDict = {
            "1d":"5m",
            "5d":"30m",
            "1mo":"1d",
            "3mo":"1d",
            "6mo":"1d"
        }

        sampleSymbols = ["BTC-USD", "ETH-USD", "XRP-USD", "TRX-USD", "MATIC-USD", "ADA-USD",
                   "SHIB-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "USDC-USD", "USDT-USD"]
        
        period = data["period"]
        originalSymbols = data["symbols"]
        interval = intervalDict[period]
        
        symbolToExtract = copy.deepcopy(originalSymbols)
        
        #adding a 2nd symbol
        if len(symbolToExtract) == 1:
            for symbol in sampleSymbols:
                if symbol not in symbolToExtract:
                    symbolToExtract.append(symbol)
                    break

        tickers = yf.Tickers(symbolToExtract)
        
        historical_data = tickers.history(period=period,interval=interval)
        
        formattedData = {}
        for symbol in originalSymbols:
            
            df = historical_data["Close"][symbol].fillna(0)
            
            dataDict = df.to_dict()
            timestampList = []
            priceList = []            
            
            for timestamp, value in dataDict.items():
                timestampList.append(pd.Timestamp(timestamp))
                priceList.append(value)
            
            formattedData[symbol] = {"date": timestampList, "price": priceList}

        return {"markets": formattedData}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/convertCurrency")
async def convertCurrency(data:dict):
    try:   
        cryptoSymbol = data["cryptoSymbol"]
        amount = data["amount"]
        
        currencyData = getCurrencyConversionData()

        if currencyData == None:
            raise HTTPException(status_code=409, detail="cannot convert currency")
        
        usdToEuro,usdToInr = currencyData
        
        crypto = yf.Ticker(cryptoSymbol)
        
        usdExchangeRate = 1
        eurExchangeRate = usdToEuro
        inrExchangeRate = usdToInr
        
        cryptoData = crypto.history(period='1d')
        cryptoPrice = cryptoData['Close'].iloc[0]

        priceInUsd = cryptoPrice * usdExchangeRate * amount
        priceInEur = cryptoPrice * eurExchangeRate * amount
        priceInInr = cryptoPrice * inrExchangeRate * amount

        return {"usd":priceInUsd,"eur":priceInEur,"inr":priceInInr}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cryptoForecast")
async def get_crypto_data(data:dict):
    crypto_name = data["cryptoSymbol"] 
    frequency = data["frequency"]
    
    global scoreDict
    scoreDict = {}

    # crypto_name = "ETH-USD"  #BTC, ETH, USDT, BNB
    # frequency = "daily"  # Change to "daily", "monthly" or "yearly" as needed
    time_format = "%Y-%m-%d"
    if frequency == "daily":
        forecast_days = 1
    elif frequency == "monthly":
        forecast_days = 10
  

    resultDF = LSTM_forecasting(crypto_name, time_format, frequency, forecast_days)
    
    df_filled = resultDF.fillna('N/A')
    
    historicalDf = df_filled[df_filled['LSTM_Frc'] == 'N/A'][['ds', 'y']]
    forecastedDf = df_filled[df_filled['y'] == 'N/A'][['ds', 'LSTM_Frc']]
    
    dateOfHistoricalDf = list(np.array(historicalDf["ds"]))
    priceOfHistoricalDf = list(np.array(historicalDf["y"]))
    
    
    dateOfForecastedDf = list(np.array(forecastedDf["ds"]))
    priceOfForecastedDf = list(np.array(forecastedDf["LSTM_Frc"]))
    
    dateOfForecastedDf.insert(0, dateOfHistoricalDf[-1])
    priceOfForecastedDf.insert(0, priceOfHistoricalDf[-1])
    
    return {"date" : [dateOfHistoricalDf,dateOfForecastedDf],"price":[priceOfHistoricalDf,priceOfForecastedDf]}