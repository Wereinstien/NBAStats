
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from models.data import create_data_and_labels
from models.data import split_train_and_test

from models.log_reg import train_log_reg_model
from models.log_reg import standardization


def train_bagging_model(X, y):
    bagg = BaggingClassifier(base_estimator=LogisticRegression(penalty='l2', max_iter=1000), n_estimators=10)
    bagg.fit(X, y)
    return bagg


def train_boosting_model(X, y):
    boost = AdaBoostClassifier(base_estimator=LogisticRegression(penalty='l2', max_iter=1000), n_estimators=25)
    boost.fit(X, y)
    return boost


def main():
    data, labels = create_data_and_labels()
    data_scaled = standardization(data)

    X_train, X_test, y_train, y_test = split_train_and_test(data_scaled, labels, size=0.3)
    bagg_model = train_bagging_model(X_train, y_train)
    boost_model = train_boosting_model(X_train, y_train)

    # 0.6665140764476997
    print(f'The logistic regression model using bagging ensemble had an accuracy of '
          f'{bagg_model.score(X_test, y_test)} on predicting the position of players.')

    # 0.5793087663080797
    print(f'The logistic regression model using boosting ensemble had an accuracy of '
          f'{boost_model.score(X_test, y_test)} on predicting the position of players.')


if __name__ == '__main__':
    main()
