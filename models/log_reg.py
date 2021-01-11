
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from models.data import create_data_and_labels
from models.data import split_train_and_test


def standardization(X):
    X_scaled = preprocessing.scale(X)
    return X_scaled


def train_log_reg_model(X, y):
    log_reg = LogisticRegression(penalty='l2', max_iter=1000)
    log_reg.fit(X, y)
    return log_reg


def main():
    data, labels = create_data_and_labels()
    data_scaled = standardization(data)

    X_train, X_test, y_train, y_test = split_train_and_test(data, labels, size=0.3)
    X_train_s, X_test_s, y_train_s, y_test_s = split_train_and_test(data_scaled, labels, size=0.3)

    model = train_log_reg_model(X_train, y_train)
    model_s = train_log_reg_model(X_train_s, y_train_s)

    # produces convergence warning
    # 0.6575875486381323
    print(f'The logistic regression model without using standardized data had an accuracy of '
          f'{model.score(X_test, y_test)} on predicting the position of players.')

    # 0.658960860608835
    print(f'The logistic regression model using standardized data had an accuracy of '
          f'{model_s.score(X_test_s, y_test_s)} on predicting the position of players.')


if __name__ == '__main__':
    main()
