import pandas
from datetime import datetime, timedelta
from tqdm import tqdm
import time
from datetime import date
import sklearn
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as stockapi

#Calls the API we will be getting data from
aps  = stockapi.REST(key_id = 'PKGV225LB530LUK2HF6H', secret_key = 'Bl2J4eBCOWtJhkWc49krrCAHblaB71s9VHcEtTiW', base_url = 'https://paper-api.alpaca.markets')

#Breaks the period into smaller sets so that we can get more data
def break_period(start,end,step_days):
    start_date = start
    delta = timedelta(days=step_days)
    date_list = []

    while end > (start_date + delta):
        date_list.append((start_date,start_date+delta))
        start_date += delta
        date_list.append((start_date,end))
    return date_list

def format_timestep_list(list):
    for i, d in enumerate(list):
        list[i] = (d[0].isoformat() + '-04:00', (d[0].isoformat().split('T')[0] + 'T23:00:00-04:00'))
    return list 

#Takes the barset and creates a dataframe out of it
def get_df(barset):
    df_rows = []
    for symbol, bar in barset.items():
        rows = bar.__dict__.get('_raw')

        for i, row in enumerate(rows):
            row['symbol'] = symbol

        df_rows.extend(rows)
    return pandas.DataFrame(df_rows)

#Downloads the data from the API and puts it in a CVS file
def download(aps, symbols, start, end, filename = 'StockData.csv'):
    steps = format_timestep_list(break_period(start,end,20))
    dataframe = pandas.DataFrame()

    for steps in tqdm(steps):
        barset = aps.get_barset(symbols, '15Min', limit = 1000, start = steps[0], end = steps[1])
        dataframe = dataframe.append(get_df(barset))
        time.sleep(0.1)

    dataframe.to_csv(filename)

#Predicts price based on the forecast.
def predict():
    #Define the name of the file and the start and end times
        StockData = 'StockData.csv'
        today = datetime.now()
        start = datetime(2020,1,1)

        while True:
            #Let the user define variables (TBA)
            symbol = input("Enter the symbol of the company: ")
            userForecast = input("How many days into the future do you want to predict?: ")

            #Downloads the data and makes a dataframe
            download(aps=aps, symbols=[symbol], start = start, end = today, filename = StockData)
            StockDatabase = pandas.read_csv('StockData.csv')

            try:
                Forecast = int(userForecast)
            except ValueError:
                print("Please type in a whole number for the forecast.")
                continue

            try:
                StockDatabase['prediction'] = StockDatabase['c'].shift(-Forecast)
            except:
                print("That symbol that you entered does not exist.")
                continue

            break

        #Creates and formats the data
        DataDatabase = StockDatabase.drop(['prediction'],1)
        data = numpy.array(DataDatabase.drop(['symbol'],1))
        data = data[:-Forecast]

        #Creates and formats the target
        target = numpy.array(StockDatabase['prediction'])
        target = target[:-Forecast]

        #Splits the data and targets to train and test the model
        data_train, data_test, target_train, target_test = train_test_split(data,target,random_state = 1, train_size = 0.7, test_size = 0.3)

        #Creates and tests the database
        stockPredict = LinearRegression().fit(data_train,target_train)
        pred = stockPredict.predict(data_test)
        print("Mean Squared Error: ", mean_squared_error(target_test, pred))
        print("Test Score: ", stockPredict.score(data_test,target_test))

        #Makes and prints a prediction forecast
        StockDatabase = StockDatabase.drop(['symbol'],1)
        data_forecast = numpy.array(StockDatabase.drop(['prediction'],1))[-Forecast:]

        stockPred = stockPredict.predict(data_forecast)
        print("Prediction Forecast: ", stockPred)

def compare():
     #Define the name of the file and the start and end times
        StockData1 = 'StockData1.csv'
        StockData2 = 'StockData2.csv'

        today = datetime.now()
        start = datetime(2020,1,1)

        while True:
            #Let the user define variables
            symbol1 = input("Enter the symbol of the first company: ")
            symbol2 = input("Enter the symbol of the second company: ")
            Forecast = 1

            #Downloads the data and makes a dataframe for the first company
            download(aps=aps, symbols=[symbol1], start = start, end = today, filename = StockData1)
            StockDatabase1 = pandas.read_csv(StockData1)

            try:
                StockDatabase1['prediction'] = StockDatabase1['c'].shift(-Forecast)
            except:
                print("That symbol that you entered does not exist.")
                continue
            
            #Downloads the data and makes a dataframe for the second company
            download(aps=aps, symbols=[symbol2], start = start, end = today, filename = StockData2)
            StockDatabase2 = pandas.read_csv(StockData2)

            try:
                StockDatabase2['prediction'] = StockDatabase2['c'].shift(-Forecast)
            except:
                print("That symbol that you entered does not exist.")
                continue

            break
         
        #Creates and formats the data for the first dataset
        DataDatabase1 = StockDatabase1.drop(['prediction'],1)
        data1 = numpy.array(DataDatabase1.drop(['symbol'],1))
        data1 = data1[:-Forecast]

        #Creates and formats the target for the first dataset
        target1 = numpy.array(StockDatabase1['prediction'])
        target1 = target1[:-Forecast]
        
        #Splits the data and targets to train and test the first model
        data_train1, data_test1, target_train1, target_test1 = train_test_split(data1,target1,random_state = 1, train_size = 0.7, test_size = 0.3)

        #Creates and predicts the first forecast
        stockPredict1 = LinearRegression().fit(data_train1,target_train1)
        StockDatabase1 = StockDatabase1.drop(['symbol'],1)
        data_forecast1 = numpy.array(StockDatabase1.drop(['prediction'],1))[-Forecast:]
        prediction1 = stockPredict1.predict(data_forecast1)

        #Creates and formats the data for the second dataset
        DataDatabase2 = StockDatabase2.drop(['prediction'],1)
        data2 = numpy.array(DataDatabase2.drop(['symbol'],1))
        data2 = data2[:-Forecast]

        #Creates and formats the target for the second dataset
        target2 = numpy.array(StockDatabase2['prediction'])
        target2 = target2[:-Forecast]

        #Splits the data and targets to train and test the first model
        data_train2, data_test2, target_train2, target_test2 = train_test_split(data2,target2,random_state = 1, train_size = 0.7, test_size = 0.3)

         #Creates and predicts the first forecast
        stockPredict2 = LinearRegression().fit(data_train2,target_train2)
        StockDatabase2 = StockDatabase2.drop(['symbol'],1)
        data_forecast2 = numpy.array(StockDatabase2.drop(['prediction'],1))[-Forecast:]
        prediction2 = stockPredict2.predict(data_forecast2)

        if prediction1 == prediction2 :
            print("The prices will be the same tommorow.")
        elif prediction1 < prediction2 :
            print("The second company will have the higher price tommorow.")
        elif prediction1 > prediction2 :
            print("The first company will have the higher price tommorow.")

Run = True

while (Run):
    while True:
        UserChoice = input("Type predict to predict a stock price, or compare to compare two future stock prices: ")
    
        if UserChoice != "Predict" and UserChoice != "predict" and UserChoice != "Compare" and UserChoice != "compare":
            print("I am sorry, but that response is invalid.")
            continue
        else:
            break

    if UserChoice == "Predict" or UserChoice == "predict":
        RunPrediction = True
        RunComparison = False
    elif UserChoice == "Compare" or UserChoice == "compare":
        RunPrediction = False
        RunComparison = True

    if (RunPrediction):
       predict()
    
    elif RunComparison:
       compare()

    #Asks the user if they want to make another prediction
    while True:
        print("Would you like do anything else? Y = Yes, N = No")
        Response = input()
    
        if Response != "Y" and Response != "N" and Response != "y" and Response != "n":
            print("I am sorry, but that response is invalid.")
            continue
        else:
            break

    if(Response == "Y" or Response == "y" ):
        Run = True
    elif(Response == "N" or Response == "n" ):
        Run = False


print("Goodbye.")


#References
#Argerich.M.F. (2020, June 16). How to Access Stock Market Data for Machine Learning in Python. Towards Data Science. https://towardsdatascience.com/how-to-access-stocks-market-data-for-machine-learning-on-python-c69db51e7a0d
#randerson112358. (2019, June 12). Predict Stock Prices Using Python and Machine Learning. Medium. https://randerson112358.medium.com/predict-stock-prices-using-python-machine-learning-53aa024da20a
