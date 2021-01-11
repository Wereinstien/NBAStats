
import pandas as pd
from sklearn.model_selection import train_test_split


SEASONS_STATS_1982_ONWARDS_CSV = 'data/Seasons_Stats_1982_Onwards.csv'


def create_data_and_labels():
    df = pd.read_csv(SEASONS_STATS_1982_ONWARDS_CSV)
    df_no_na = df[[col for col in df.columns if col not in ['blank', 'blank2']]].dropna()

    X = df_no_na[df_no_na.columns[6:]]
    y = df_no_na['Pos']

    return X, y


def split_train_and_test(X, y, size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    return X_train, X_test, y_train, y_test
