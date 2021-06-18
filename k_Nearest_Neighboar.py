import pandas as pd
import numpy as np
import time

# to eliminate error -> 'A value is trying to be set on a copy of a
# slice from a DataFrame'
pd.options.mode.chained_assignment = None

# import data as Dataframe Pandas
df = pd.read_csv('Diabetes.csv')


def fill_zero_values(df):
    for i in df.iloc[:, :8]: df[str(i)].replace(to_replace=0, value=df[str(i)].mean(), inplace=True)


def minmax(dataset):
    for i in dataset:
        min_val = dataset[str(i)].min()
        max_val = dataset[str(i)].max()
        for j in range(dataset.shape[0]):
            dataset[str(i)].iloc[j] = (dataset[str(i)].iloc[j] - min_val) / (max_val - min_val)


def euclidean(data_test, data_train):
    return np.sqrt(np.sum((np.subtract(data_test, data_train)) ** 2, axis=1))


def calculate_distance(test_data, train_data):
    temp = np.zeros((train_data.shape[0], 2))
    temp[:, 0] = euclidean(test_data, train_data.iloc[:, :8])
    temp[:, 1] = train_data['Outcome']
    return temp


def min_distance(k, temp):
    return temp[np.argpartition(temp, kth=(k - 1), axis=0, )[:k, 0]]


def get_most_occurrences(calon_chosen):
    return np.bincount(calon_chosen[:, 1].astype('int32')).argmax()


def classification(k, test_data, train_data):
    for i in range(test_data.shape[0]):
        distance = calculate_distance(test_data.iloc[i, :8], train_data)
        test_data['Outcome'].iloc[i] = get_most_occurrences(min_distance(k, distance))


def calculate_accuracy(test_data, accuracy_data):
    temp = 0
    for i in range(test_data.shape[0]):
        if test_data["Outcome"].iloc[i] == accuracy_data['Outcome'].iloc[i]:
            temp += 1
    return (temp / test_data.shape[0]) * 100


def main():
    print("Preprocessing Data...")
    fill_zero_values(df)
    minmax(df)
    print("Preprocessing Data finish")

    # Declaration of every fold
    train_fold1 = df.iloc[:614].copy()
    test_fold1 = df.iloc[614:].copy()
    test_acc_fold1 = df.iloc[614:]

    train_fold2 = pd.concat([df.iloc[:461], df.iloc[615:768]])
    test_fold2 = df.iloc[461:615].copy()
    test_acc_fold2 = df.iloc[461:615]

    train_fold3 = pd.concat([df.iloc[:307], df.iloc[461:768]])
    test_fold3 = df.iloc[307:461].copy()
    test_acc_fold3 = df.iloc[307:461]

    train_fold4 = pd.concat([df.iloc[:154], df.iloc[308:768]])
    test_fold4 = df.iloc[154:308].copy()
    test_acc_fold4 = df.iloc[154:308]

    train_fold5 = df.iloc[154:768].copy()
    test_fold5 = df.iloc[:154].copy()
    test_acc_fold5 = df.iloc[:154]

    best_k = 0
    best_avg_acc = 0
    best_avg_folds = []

    start = time.time()

    # looping to finds the best k value with a limit of 50 iterations
    print("Finding the best k value (process takes approx. 1.5 minute)...")
    for i in range(50):
        # makes k an odd value
        k = 2 * i + 1

        classification(k, test_fold1, train_fold1)
        classification(k, test_fold2, train_fold2)
        classification(k, test_fold3, train_fold3)
        classification(k, test_fold4, train_fold4)
        classification(k, test_fold5, train_fold5)

        acc_fold1 = calculate_accuracy(test_fold1, test_acc_fold1)
        acc_fold2 = calculate_accuracy(test_fold2, test_acc_fold2)
        acc_fold3 = calculate_accuracy(test_fold3, test_acc_fold3)
        acc_fold4 = calculate_accuracy(test_fold4, test_acc_fold4)
        acc_fold5 = calculate_accuracy(test_fold5, test_acc_fold5)

        avg_acc = (acc_fold1 + acc_fold2 + acc_fold3 + acc_fold4 + acc_fold5) / 5

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_k = k
            best_avg_folds = [acc_fold1, acc_fold2, acc_fold3, acc_fold4, acc_fold5]

    print('The best k value:', best_k)
    print('Accuracy on Fold 1:', round(best_avg_folds[0], 2), "%")
    print('Accuracy on Fold 2:', round(best_avg_folds[1], 2), "%")
    print('Accuracy on Fold 3:', round(best_avg_folds[2], 2), "%")
    print('Accuracy on Fold 4:', round(best_avg_folds[3], 2), "%")
    print('Accuracy on Fold 5:', round(best_avg_folds[4], 2), "%")
    print('Average Accuracy:', round(best_avg_acc, 2), "%")
    end = time.time()
    print('Process time for finding best k:', round(end - start, 2),'s')


main()
