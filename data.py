import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data():
    df = pd.read_csv('/Users/saja/Desktop/workout detector/coords.csv')

    X = df.drop('class', axis=1) # features
    y = df['class'] # target value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    return X_train, X_test, y_train, y_test
