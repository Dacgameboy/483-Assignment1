import time as t
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from question import Question

# class Question(Enum):
#     FeatureScale = "fs"
#     Regular = "df"
#     Gradient = "gd"

class LinRegr:

    def __init__(self, filename, goal_order):

        self.goal_order = goal_order + 1

        # Initialize df and N (the length of df)
        self.read_file(filename)

        # Separate the df into training and testing sets
        self.sample_sets()

        # Original features
        self.orig_col_names = list(self.df.columns[:-1])

        # Current order of the regression function
        self.order = 1

        self.order_fs = 1

        self.do_feature_scale()

    # This function adds a new column for each ORIGINAL column that existed in the dataset
    # with raised to a power of "order"
    #
    # df - Data frame
    # data_len - Number of examples (N)
    # orig_col_names - A list of the original feature names
    # order - The order to increase to
    def inc_order(self, problem):

        match problem:

            case Question.FeatureScale:

                train_len = len(self.training_x_fs)
                test_len = len(self.testing_x_fs)

                self.order_fs += 1

                for i in range(len(self.orig_col_names)):

                    # The data for the new column
                    training_col_data = []
                    testing_col_data = []

                    for example in range(train_len):

                        #new_col_data.append(df[col_name][example] ** order)
                        #print(self.training_x_fs[example])
                        #print(example)
                        #print(i)
                        training_col_data.append(self.training_x_fs[example][i] ** self.order_fs)

                    for example in range(test_len):
                    
                        testing_col_data.append(self.testing_x_fs[example][i] ** self.order_fs)

                    # Create the name for the column
                    #name = str(i) + " Order: " + str(self.order)

                    # Insert the column
                    #df.insert(0, name, new_col_data)

                    #self.training_x_fs = np.append(self.training_x_fs, [training_col_data], axis=1)
                    self.training_x_fs = np.insert(self.training_x_fs, len(self.training_x_fs[0]), [training_col_data], axis=1)

                    #self.testing_x_fs = np.append(self.testing_x_fs, [testing_col_data], axis=1)
                    self.testing_x_fs = np.insert(self.testing_x_fs, len(self.testing_x_fs[0]), [testing_col_data], axis=1)

            case Question.LinReg:

                train_len = len(self.training_x)
                test_len = len(self.testing_x)

                self.order += 1

                for i in range(len(self.orig_col_names)):

                    # The data for the new column
                    training_col_data = []
                    testing_col_data = []

                    for example in range(train_len):

                        #new_col_data.append(df[col_name][example] ** order)
                        training_col_data.append(self.training_x[example][i] ** self.order)

                    for example in range(test_len):
                    
                        testing_col_data.append(self.testing_x[example][i] ** self.order)

                    # Create the name for the column
                    #name = str(i) + " Order: " + str(self.order)

                    # Insert the column
                    #df.insert(0, name, new_col_data)
                    
                    self.training_x = np.insert(self.training_x, len(self.training_x[0]), [training_col_data], axis=1)

                    self.testing_x = np.insert(self.testing_x, len(self.testing_x[0]), [testing_col_data], axis=1)

                #print("Training Row: " + str(self.training_x[0]))
                #print("Original Number: " + str(self.training_x[0][0]))
                #print("New Number (squared of original number): " + str(self.training_x[0][4]))

        #return df

        # Separate the data into features
        # x = df.loc[:, df.columns != "Idx"]

        # Split the training and testing data into two parts
        # training_x = x.iloc[:-(int(N / 4))]
        # testing_x = x.iloc[-(int(N / 4)):]

        # Separate the data into classifiers
        # y = df.loc[:, df.columns == "Idx"]

    def read_file(self, filename):

        self.df = pd.read_csv(filename)

        self.N = len(self.df)


    def sample_sets(self):

        # Take first sample (training)
        self.df_new = self.df.sample(frac = 0.75, replace=True)

        # Remove classifier
        self.training_x = self.df_new.loc[:, self.df.columns != "Idx"]

        # Remove features
        self.training_y = self.df_new.loc[:, self.df.columns == "Idx"]

        # Take second sample (testing)
        self.df_new = self.df.sample(frac = 0.25, replace=True)

        # Remove classifier
        self.testing_x = self.df_new.loc[:, self.df.columns != "Idx"]

        # Remove features
        self.testing_y = self.df_new.loc[:, self.df.columns == "Idx"]


        self.training_x.reset_index(drop=True, inplace=True)
        self.testing_x.reset_index(drop=True, inplace=True)

        self.training_x = pd.DataFrame.to_numpy(self.training_x)
        self.training_y = pd.DataFrame.to_numpy(self.training_y)

        self.testing_x = pd.DataFrame.to_numpy(self.testing_x)
        self.testing_y = pd.DataFrame.to_numpy(self.testing_y)

    
    def do_feature_scale(self):

        self.scaler = MinMaxScaler()

        self.training_x_fs = self.scaler.fit_transform(self.training_x)
        self.testing_x_fs = self.scaler.fit_transform(self.testing_x)

        #self.training_y_fs = self.scaler.fit_transform(self.training_y)
        #self.testing_y_fs = self.scaler.fit_transform(self.testing_y)

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

    def train(self, problem):

        match problem:

            case Question.FeatureScale:

                for i in range(1, self.goal_order):

                    if self.order_fs < i:
                    
                        self.inc_order(problem)

                    # Create the linear regression
                    self.regr = linear_model.LinearRegression()
                    StartTime = t.time()
                    self.regr.fit(self.training_x_fs, self.training_y)
                    EndTime = t.time()
                    self.predict_y = self.regr.predict(self.testing_x_fs)
                    print("Intercept: \n", self.regr.intercept_)
                    print("\nCoefficients: \n", self.regr.coef_)

                    self.predict_y_2 = self.regr.predict(self.training_x_fs)

                    print("\nErrors for testing data: Order " + str(i))

                    print("Root Mean Squared Error: \n", mse(self.testing_y, self.predict_y, squared=False))

                    print("Coefficient of determination: \n", r2_score(self.testing_y, self.predict_y))


                    print("\nErrors for training data: Order " + str(i))

                    print("Root Mean Squared Error: \n", mse(self.training_y, self.predict_y_2, squared=False))

                    print("Coefficient of determination: \n", r2_score(self.training_y, self.predict_y_2))

                    print("Total Time: " + str(EndTime - StartTime))

                    print()

            case Question.LinReg:

                for i in range(1, self.goal_order):

                    if self.order < i:
                    
                        self.inc_order(problem)

                    # Create the linear regression
                    self.regr = linear_model.LinearRegression()
                    StartTime = t.time()
                    self.regr.fit(self.training_x, self.training_y)
                    EndTime = t.time()
                    self.predict_y = self.regr.predict(self.testing_x)
                    print("Intercept: \n", self.regr.intercept_)
                    print("\nCoefficients: \n", self.regr.coef_)

                    self.predict_y_2 = self.regr.predict(self.training_x)

                    print("\nErrors for testing data: Order " + str(i))

                    print("Root Mean Squared Error: \n", mse(self.testing_y, self.predict_y, squared=False))

                    print("Coefficient of determination: \n", r2_score(self.testing_y, self.predict_y))


                    print("\nErrors for training data: Order " + str(i))

                    print("Root Mean Squared Error: \n", mse(self.training_y, self.predict_y_2, squared=False))

                    print("Coefficient of determination: \n", r2_score(self.training_y, self.predict_y_2))

                    print("Total Time: " + str(EndTime - StartTime))

                    print()
"""
for col in testing_x:
plt.scatter(testing_x[col], testing_y, color="black")
plt.plot(testing_x[col], predict_y, color="blue", linewidth=3)
plt.xticks()
plt.yticks()
plt.xlabel(col)
plt.ylabel("Idx")
plt.show()
"""

if __name__ == "__main__":
    
    # Initialize the object
    linRegr = LinRegr("Data1.csv", 20)
    #gradient_descent = gd("Data1.csv")

    myProblems = [Question.FeatureScale]

    for Problem in myProblems:

        match Problem:

            case Question.LinReg:

                linRegr.train(Problem)
                #print("Lin Regr")

            case Question.Gradient:

                #print(gradient_descent.train())
                print("Gradient Descent Here...")

            case Question.FeatureScale:

                linRegr.train(Problem)
                #print("Feature Scaling")

# Order 2 output:

# w: [54.04725842  1.42802964 -0.38945084 -0.08894312 -1.43481346 -0.07986536
#  -0.04123719 -0.01299227  0.05938129]
# Training RMSE: 0.12281222515401258
# Training Coefficients of Determination: 0.7621873038665664
# Training RMSE (Library): 0.12281222515401125
# X vals: [[ 1.          0.29276967  1.87749057 ...  3.52497083  0.67216995
#    0.26081345]
#  [ 1.          0.87830902 -0.95606143 ...  0.91405347  0.37087827
#    1.11800205]
#  [ 1.         -0.87830902  0.82802686 ...  0.68562849  0.17849325
#    1.32375665]
#  ...
#  [ 1.         -1.46384837 -0.95606143 ...  0.91405347  3.24894808
#    1.08495611]
#  [ 1.          0.29276967 -0.22143684 ...  0.04903427  0.00318981
#    0.19138192]
#  [ 1.         -0.87830902  1.87749057 ...  3.52497083  0.0426371
#    2.16908047]]
# Training_y_numpy: [54.2796828]
# Predict_y: 54.24853774629539
# Training Coefficient of determination (Library): 0.7621872907668055
# Total Time: 534.3243877887726 seconds