import random

import pandas as pd
import numpy as np
from datetime import datetime

import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
def extract_Total_RE(eskom_dataframe, dictOfDates):
    for index, row in eskom_dataframe.iterrows():
        # Accessing values
        column1_value = row['Date']
        date_obj = datetime.strptime(column1_value, '%A, %d %B %Y')
        formatted_date = date_obj.strftime('%Y/%m/%d')
        if formatted_date in dictOfDates:
            tRE = dictOfDates[formatted_date]
            temp = tRE + row['Total_RE']
            dictOfDates[formatted_date] = temp
    return dictOfDates


def analyize_Weather(weather_df):
    DictionaryOfWeather = {}
    for index, row in weather_df.iterrows():
        DateTime = row['datetime']
        date_obj = datetime.strptime(DateTime, '%Y/%m/%d')

        formatted_date = date_obj.strftime('%Y/%m/%d')
        DictionaryOfWeather[formatted_date] = 0

    return DictionaryOfWeather

def preprocess_weather(weather_df, dictionaryOfDates, theTotalRE):
    #Here I will remove all the unnecassiry columns
    #And binary encode certain values
    weather_df.drop('conditions', axis=1, inplace=True)
    weather_df.drop('description', axis=1, inplace=True)
    weather_df.drop('stations', axis=1, inplace=True)
    weather_df.drop('sunrise', axis=1, inplace=True)
    weather_df.drop('sunset', axis=1, inplace=True)
    weather_df.drop('preciptype', axis=1, inplace=True)

    columns_to_drop = [col for col in weather_df.columns if 'Unnamed' in col]

    # Dropping the columns
    weather_df.drop(columns=columns_to_drop, inplace=True)
    icon_mapping = {
        'clear-day': 0,
        'rain': 1,
        'wind': 2,
        'partly-cloudy-day': 3,
        'cloudy': 4
    }
    weather_df['icon'] = weather_df['icon'].replace(icon_mapping)
    average_locations(weather_df,dictionaryOfDates,theTotalRE)
def average_locations(weather_df, dictionaryOfDates,theTotalRE):
    finalVersion = pd.DataFrame()
    for x in dictionaryOfDates:
        results = weather_df.loc[weather_df["datetime"] == x]
        # Avoid SettingWithCopyWarning by creating a new DataFrame directly
        results = results.drop('name', axis=1)
        results = results.drop('datetime', axis=1)
        # Select only numerical columns for calculating the mean
        # Select only numerical columns except 'icon' for calculating the mean
        numerical_cols = results.select_dtypes(include=['number'])

        # Calculating the average for each numerical field except 'icon'
        averages = numerical_cols.mean()

        # Calculating the median for 'icon'
        icon_median = results['icon'].mode()

        # Creating a new DataFrame with the averages
        average_df = averages.to_frame().transpose()

        # Adding the median of 'icon' to the DataFrame
        average_df['icon'] = icon_median
        average_df['Total_RE'] = theTotalRE[x]
        finalVersion = pd.concat([finalVersion, average_df], ignore_index=True)

    print(finalVersion)

def readInputFile(pathForWeather, pathForEskom):
    dfweather = pd.read_csv(pathForWeather)
    dfeskomData = pd.read_csv(pathForEskom)
    dict_of_dates = analyize_Weather(dfweather)
    theTotalRE = extract_Total_RE(dfeskomData,dict_of_dates)
    preprocess_weather(dfweather,dict_of_dates,theTotalRE)



from sklearn.model_selection import train_test_split


feature_scaler = StandardScaler()
target_scaler = StandardScaler()

def messAroundWithData(df):
    # Normalize features
    features = df.drop(['Total_RE', 'moonphase', 'icon'], axis=1)
    normalized_features = pd.DataFrame(feature_scaler.fit_transform(features), columns=features.columns)

    # Normalize target variable
    target = df[['Total_RE']]
    normalized_target = target_scaler.fit_transform(target)

    # Combine normalized features and target
    df_scaled = pd.concat([normalized_features, pd.DataFrame(normalized_target, columns=['Total_RE'])], axis=1)
    return df_scaled

import matplotlib.pyplot as plt
def runCode():
    df = pd.read_csv('processed.csv')
    df = messAroundWithData(df)
    X = df.drop('Total_RE', axis=1).values
    y = df['Total_RE'].values

    # Convert arrays to PyTorch tensors
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate the index for splitting the data
    split_index = int(len(X) * 0.8)

    # Split the data into training and testing sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the model
    class RegressionNN(nn.Module):
        # def __init__(self):
        #     super(RegressionNN, self).__init__()
        #     self.fc1 = nn.Linear(X.shape[1], 64)
        #     self.af1 = nn.ReLU()
        #     self.fc2 = nn.Linear(64, 32)
        #     self.af2 = nn.ReLU()
        #     self.fc3 = nn.Linear(32, 12)
        #     self.af3 = nn.ReLU()
        #     self.fc4 = nn.Linear(12, 1)  # Adjusted the output dimension
        #
        # def forward(self, x):
        #     x = self.af1(self.fc1(x))
        #     x = self.af2(self.fc2(x))
        #     x = self.af3(self.fc3(x))
        #     x = self.fc4(x)  # Added the last layer
        #     return x
        def __init__(self):
            super(RegressionNN, self).__init__()
            self.fc1 = nn.Linear(X.shape[1], 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.af1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.25)
            self.fc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.af2 = nn.ReLU()
            self.drop2 = nn.Dropout(0.25)
            self.fc3 = nn.Linear(32, 12)
            self.bn3 = nn.BatchNorm1d(12)
            self.af3 = nn.ReLU()
            self.drop3 = nn.Dropout(0.25)
            self.fc4 = nn.Linear(12, 1)

        def forward(self, x):
            x = self.drop1(self.af1(self.bn1(self.fc1(x))))
            x = self.drop2(self.af2(self.bn2(self.fc2(x))))
            x = self.drop3(self.af3(self.bn3(self.fc3(x))))
            x = self.fc4(x)
            return x

    model = RegressionNN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(45):
        for batch_idx, (data, target) in enumerate(data_loader):
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Compute the loss
            loss = criterion(output, target)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Evaluate the model on test data
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        predictions_np = predictions.numpy()  # Convert to numpy
        denormalized_predictions = target_scaler.inverse_transform(predictions_np)

        actual_np = y_test_tensor.numpy()  # Convert to numpy
        denormalized_actual = target_scaler.inverse_transform(actual_np)

        print(f"Test Loss: {test_loss.item()}")

        df_predictions = pd.DataFrame(denormalized_predictions)
        df_actual = pd.DataFrame(denormalized_actual)

        # Save the DataFrames to CSV files
        df_predictions.to_csv('predictions.csv', index=False)
        df_actual.to_csv('actual.csv', index=False)


if __name__ == '__main__':
    # readInputFile('WeatherData.csv','EskomData.csv')
    random.seed(45)
    torch.manual_seed(45)

    runCode()