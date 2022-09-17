import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score
import pandas as pd

# This function adds a new column for each ORIGINAL column that existed in the dataset
# with raised to a power of "order"
#
# df - Data frame
# data_len - Number of examples (N)
# orig_col_names - A list of the original feature names
# order - The order to increase to
def inc_order(training_x, testing_x, orig_col_names, order):

    train_len = len(training_x)
    test_len = len(testing_x)

    for col_name in orig_col_names:

        # The data for the new column
        training_col_data = []
        testing_col_data = []

        for example in range(train_len):

            #new_col_data.append(df[col_name][example] ** order)
            training_col_data.append(training_x[col_name][example] ** order)

        for example in range(test_len):
        
            testing_col_data.append(testing_x[col_name][example] ** order)

        # Create the name for the column
        name = col_name + " Order: " + str(order)

        # Insert the column
        #df.insert(0, name, new_col_data)

        training_x.insert(0, name, training_col_data)

        testing_x.insert(0, name, testing_col_data)

    #return df





df = pd.read_csv("Data1.csv")

N = len(df)


# Original features
orig_col_names = list(df.columns[:-1])

# Current order of the regression function
order = 1

# Separate the data into features
# x = df.loc[:, df.columns != "Idx"]

# Split the training and testing data into two parts
# training_x = x.iloc[:-(int(N / 4))]
# testing_x = x.iloc[-(int(N / 4)):]

# Separate the data into classifiers
# y = df.loc[:, df.columns == "Idx"]

# Take first sample (training)
df_new = df.sample(frac = 0.75, replace=True)

# Remove classifier
training_x = df_new.loc[:, df.columns != "Idx"]

# Remove features
training_y = df_new.loc[:, df.columns == "Idx"]

# Take second sample (testing)
df_new = df.sample(frac = 0.25, replace=True)

# Remove classifier
testing_x = df_new.loc[:, df.columns != "Idx"]

# Remove features
testing_y = df_new.loc[:, df.columns == "Idx"]


training_x.reset_index(drop=True, inplace=True)
testing_x.reset_index(drop=True, inplace=True)

#print(df)
#print(df_new)
#print(training_x)
#print(testing_x)


# # Split the training and testing data into two parts
# training_y = y.iloc[:-(int(N / 4))]
# testing_y = y.iloc[-(int(N / 4)):]

# data.sample(20,random_state=1)

#training_y = y.iloc[:int(N * 0.8)]
#testing_y = y.iloc[:int(N * 0.2)]

#print(training_x)
#print(testing_x)

#print(training_y)
#print(testing_y)

for i in range(5):

    # Create the linear regression
    regr = linear_model.LinearRegression()

    regr.fit(training_x, training_y)

    predict_y = regr.predict(testing_x)
    print("Intercept: \n", regr.intercept_)
    print("\nCoefficients: \n", regr.coef_)

    predict_y_2 = regr.predict(training_x)

    print("\nErrors for testing data: Order " + str(order))

    print("Root Mean Squared Error: \n", mse(testing_y, predict_y, squared=False))

    print("Coefficient of determination: \n", r2_score(testing_y, predict_y))


    print("\nErrors for training data: Order " + str(order))

    print("Root Mean Squared Error: \n", mse(training_y, predict_y_2, squared=False))

    print("Coefficient of determination: \n", r2_score(training_y, predict_y_2))

    print()

    order = order + 1

    inc_order(training_x, testing_x, orig_col_names, order)






"""for col in testing_x:

    plt.scatter(testing_x[col], testing_y, color="black")
    plt.plot(testing_x[col], predict_y, color="blue", linewidth=3)
    plt.xticks()
    plt.yticks()
    plt.xlabel(col)
    plt.ylabel("Idx")
    plt.show()
"""