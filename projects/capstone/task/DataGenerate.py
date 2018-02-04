import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from task import IndicatorGenerate

def generateData():
    predict_name = 'Adj Close'

    # Read VGT data
    path = "../Data/VGT.csv"
    vgt = pd.read_csv(path)
    vgt['Date'] = pd.to_datetime(vgt['Date'])
    vgt = vgt.set_index('Date')

    # Generate indicator
    vgt = IndicatorGenerate.generateIndicator(vgt, predict_name)

    # Merge unemployment data and interest_rate data with VGT
    interest_rate_path = "../Data/InterestRate.csv"
    interest_rate = pd.read_csv(interest_rate_path)

    unemployment_rate_path = "../Data/unemploymentData.csv"
    unemployment_rate = pd.read_csv(unemployment_rate_path)

    interest_unemployment_rate = pd.merge(interest_rate, unemployment_rate, how='inner', on=['observation_date'])
    interest_unemployment_rate['observation_date'] = pd.to_datetime(interest_unemployment_rate['observation_date'])

    interest_unemployment_rate = interest_unemployment_rate.set_index("observation_date")


    interest_unemployment_rate = interest_unemployment_rate.reindex(vgt.index, method="nearest")

    vgt = vgt.join(interest_unemployment_rate, how='inner')

    # Drop rows contain N/A value
    vgt = vgt.dropna()

    vgt_target = vgt[predict_name]
    vgt_data = vgt.drop([predict_name], axis=1)
    # print(list(vgt_data))
    # print(list(vgt_data.iloc[0]))

    # 'Open', 'High', 'Low', 'Close', 'Volume'
    # vgt_data=vgt_data.drop('Open', axis=1)
    # vgt_data=vgt_data.drop('High', axis=1)
    # vgt_data=vgt_data.drop('Low', axis=1)
    # vgt_data=vgt_data.drop('Close', axis=1)
    # vgt_data=vgt_data.drop('Volume', axis=1)
    # vgt_data_log = vgt_data
    #  Normalizing Features
    vgt_data_log = pd.DataFrame(data=vgt_data)
    vgt_data_log['Volume'] = vgt_data['Volume'].apply(lambda x : np.log(x + 1))

    # print(list(vgt_data_log))
    scaler = preprocessing.StandardScaler().fit(vgt_data_log)
    vgt_data_log_transform = scaler.transform(vgt_data_log)

    X_train, X_test, y_train, y_test = train_test_split(vgt_data_log_transform, vgt_target, test_size = 0.2, random_state = 0)


    print("Training set has " + str(X_train.shape[0]) + " samples")
    print("Testing set has " + str(X_test.shape[0]) + " samples")
    return X_train, X_test, y_train, y_test


