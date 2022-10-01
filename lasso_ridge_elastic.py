import time as t
import pandas as pd
import numpy as np
from question import Question
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

# class Question(Enum):
#     Regular = "df"
#     K_Fold = "kf"

<<<<<<< HEAD
scaler = StandardScaler()
=======
>>>>>>> 66cd9b193b440e4b4bd9631d213cc0f9dd3b51ae

class LassoRidgeElastic:

    def __init__(self, filename, goal_order=2, fold_number=10):

        self.goal_order = goal_order

        self.order = 1
        self.order_kf = 1

        # Read the csv file
        self.read_file(filename)

        self.sample_sets()

        self.features = list(self.df.columns[:-1])

        self.do_K_Fold(fold_number)

    def inc_order(self, problem):

        match problem:

            case Question.K_Fold:

                X_kf_len = len(self.X_kf)
                #test_len = len(self.testing_x_fs)

                self.order_kf += 1

                for i in range(len(self.features)):

                    # The data for the new column
                    training_col_data = []
                    #testing_col_data = []

                    for example in range(X_kf_len):

                        #new_col_data.append(df[col_name][example] ** order)
                        # print(self.training_x_fs[example])
                        # print(example)
                        # print(i)
                        training_col_data.append(
                            self.X_kf[example][i] ** self.order_kf)

                    # for example in range(test_len):

                        #testing_col_data.append(self.testing_x_fs[example][i] ** self.order_kf)

                    # Create the name for the column
                    #name = str(i) + " Order: " + str(self.order)

                    # Insert the column
                    #df.insert(0, name, new_col_data)

                    #self.training_x_fs = np.append(self.training_x_fs, [training_col_data], axis=1)
                    self.X_kf = np.insert(self.X_kf, len(self.X_kf[0]), [
                                          training_col_data], axis=1)

                    #self.testing_x_fs = np.append(self.testing_x_fs, [testing_col_data], axis=1)
                    #self.testing_x_fs = np.insert(self.testing_x_fs, len(self.testing_x_fs[0]), [testing_col_data], axis=1)

            case Question.NormReg:

                train_len = len(self.X_train)
                test_len = len(self.X_test)

                self.order += 1

                for i in range(len(self.features)):

                    # The data for the new column
                    training_col_data = []
                    testing_col_data = []

                    for example in range(train_len):

                        #new_col_data.append(df[col_name][example] ** order)
                        training_col_data.append(
                            self.X_train[example][i] ** self.order)

                    for example in range(test_len):

                        testing_col_data.append(
                            self.X_test[example][i] ** self.order)

                    # Create the name for the column
                    #name = str(i) + " Order: " + str(self.order)

                    # Insert the column
                    #df.insert(0, name, new_col_data)

                    self.X_train = np.insert(self.X_train, len(
                        self.X_train[0]), [training_col_data], axis=1)

                    self.X_test = np.insert(self.X_test, len(
                        self.X_test[0]), [testing_col_data], axis=1)

    def read_file(self, filename):

        self.df = pd.read_csv(filename)

        # print(df.shape)
        # print(df)
        # print(df.describe())
        target_column = self.df[['Idx']]

        mmscaler = MinMaxScaler()
        data = pd.DataFrame.to_numpy(self.df)

        scaled_data = mmscaler.fit_transform(data)

        # print(scaled_data)

        # # split target and features
        # target_column = ['Idx']
        # features = list(set(list(self.df.columns)) - set(target_column))

        df_scaled_data = pd.DataFrame(
            scaled_data, columns=['T', 'P', 'TC', 'SV', 'Idx'])

        df_scaled_data.drop('Idx', inplace=True, axis=1)

        # print(df_scaled_data.describe())

        df_scaled_data = pd.concat(
            [df_scaled_data, target_column], axis='columns')

        # print(df_scaled_data.describe())

        self.df = df_scaled_data

        # # standardize the values
        # self.df[features] = self.df[features]/self.df[features].max()
        # print(self.df.describe())

    def sample_sets(self):

        # Take first sample (training)
        self.df_new = self.df.sample(frac=0.75, replace=True)

        # Remove classifier
        self.X_train = self.df_new.loc[:, self.df.columns != "Idx"]

        # Remove features
        self.y_train = self.df_new.loc[:, self.df.columns == "Idx"]

        # Take second sample (testing)
        self.df_new = self.df.sample(frac=0.25, replace=True)

        # Remove classifier
        self.X_test = self.df_new.loc[:, self.df.columns != "Idx"]

        # Remove features
        self.y_test = self.df_new.loc[:, self.df.columns == "Idx"]

        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)

        self.X_train = pd.DataFrame.to_numpy(self.X_train)
        self.y_train = pd.DataFrame.to_numpy(self.y_train)

        self.X_test = pd.DataFrame.to_numpy(self.X_test)
        self.y_test = pd.DataFrame.to_numpy(self.y_test)

        # X = self.df[self.features].values
        # y = self.df[self.target_column].values

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

    def do_K_Fold(self, k=0):

        self.kf = KFold(n_splits=k, shuffle=True)

        self.X_kf = self.df.loc[:, self.df.columns != "Idx"]
        self.y_kf = self.df.loc[:, self.df.columns == "Idx"]

        self.X_kf.reset_index(drop=True, inplace=True)
        self.y_kf.reset_index(drop=True, inplace=True)

        self.X_kf = pd.DataFrame.to_numpy(self.X_kf)
        self.y_kf = pd.DataFrame.to_numpy(self.y_kf)

    # print(X_train.shape)
    # print(X_test.shape)

    def train(self, Problem):

        match Problem:
            case Question.K_Fold:

                for i in range(1, self.goal_order):

                    if self.order < i:

                        self.inc_order(Problem)

                    fold_number = 1

                    for train_index, test_index in self.kf.split(self.X_kf):

                        # Split the data by index
                        self.X_train_kf, self.X_test_kf = self.X_kf[train_index], self.X_kf[test_index]
                        self.y_train_kf, self.y_test_kf = self.y_kf[train_index], self.y_kf[test_index]

                        #print("Testing X: " + str(self.X_test_kf))
                        #print("Testing Y: " + str(self.X_test_kf))

                        print("Order: " + str(i))
                        print("Fold Number: " + str(fold_number))

                        print("\nRidge:")
                        rr = Ridge(alpha=0.001)
                        rr.fit(self.X_train_kf, self.y_train_kf)
                        pred_train_rr = rr.predict(self.X_train_kf)
                        print("Intercept: " + str(rr.intercept_))
                        print("Coefficients: " + str(rr.coef_))
                        print(
                            "Training RMSE: " + str(np.sqrt(mean_squared_error(self.y_train_kf, pred_train_rr))))
                        print("Training Coefficient of Determination: " +
                              str(r2_score(self.y_train_kf, pred_train_rr)))

                        pred_test_rr = rr.predict(self.X_test_kf)
                        print(
                            "Testing RMSE: " + str(np.sqrt(mean_squared_error(self.y_test_kf, pred_test_rr))))
                        print("Testing Coefficient of Determination: " +
                              str(r2_score(self.y_test_kf, pred_test_rr)))

                        print("\nOrder: " + str(i))
                        print("Fold Number: " + str(fold_number))

                        print("\n\nLasso: ")
                        model_lasso = Lasso(alpha=0.0001, tol=1e-1)
                        model_lasso.fit(self.X_train_kf, self.y_train_kf)
                        pred_train_lasso = model_lasso.predict(self.X_train_kf)
                        print("Intercept: " + str(model_lasso.intercept_))
                        print("Coefficients: " + str(model_lasso.coef_))
                        print(
                            "Training RMSE: " + str(np.sqrt(mean_squared_error(self.y_train_kf, pred_train_lasso))))
                        print("Training Coefficient of Determination: " +
                              str(r2_score(self.y_train_kf, pred_train_lasso)))

                        pred_test_lasso = model_lasso.predict(self.X_test_kf)
                        print(
                            "Testing RMSE: " + str(np.sqrt(mean_squared_error(self.y_test_kf, pred_test_lasso))))
                        print("Testing Coefficient of Determination: " +
                              str(r2_score(self.y_test_kf, pred_test_lasso)))

                        print("\nOrder: " + str(i))
                        print("Fold Number: " + str(fold_number))

                        # Elastic Net
                        print("\n\nElastic Net: ")
                        model_enet = ElasticNet(alpha=0.0001, tol=1e-1)
                        model_enet.fit(self.X_train_kf, self.y_train_kf)
                        pred_train_enet = model_enet.predict(self.X_train_kf)
                        print("Intercept: " + str(model_enet.intercept_))
                        print("Coefficients: " + str(model_enet.coef_))
                        print(
                            "Training RMSE: " + str(np.sqrt(mean_squared_error(self.y_train_kf, pred_train_enet))))
                        print("Training Coefficient of Determination: " +
                              str(r2_score(self.y_train_kf, pred_train_enet)))

                        pred_test_enet = model_enet.predict(self.X_test_kf)
                        print(
                            "Testing RMSE: " + str(np.sqrt(mean_squared_error(self.y_test_kf, pred_test_enet))))
                        print("Testing Coefficient of Determination: " +
                              str(r2_score(self.y_test_kf, pred_test_enet)))

                        print("\n\n")

                        fold_number += 1

            case Question.NormReg:

                for i in range(1, self.goal_order):

                    if self.order < i:

                        self.inc_order(Problem)

                    print("Order: " + str(i))

                    print("\nRidge:")
<<<<<<< HEAD
                    rr = Ridge(alpha=1)
=======
                    rr = Ridge(
                        alpha=0.001)
>>>>>>> 66cd9b193b440e4b4bd9631d213cc0f9dd3b51ae
                    #rr = make_pipeline(('scl', StandardScaler(with_mean=False)), Ridge())

                    StartRTime = t.time()
                    rr.fit(self.X_train, self.y_train)
                    EndRTime = t.time()

                    pred_train_rr = rr.predict(self.X_train)

                    print(f"Ridge Training Time: {EndRTime - StartRTime:.5f}")
                    print("Intercept: " + str(rr.intercept_))
                    print("Coefficients: " + str(rr.coef_))
                    print(
                        "Training RMSE: " + str(np.sqrt(mean_squared_error(self.y_train, pred_train_rr))))
                    print("Training Coefficient of Determination: " +
                          str(r2_score(self.y_train, pred_train_rr)))

                    pred_test_rr = rr.predict(self.X_test)
                    print(
                        "Testing RMSE: " + str(np.sqrt(mean_squared_error(self.y_test, pred_test_rr))))
                    print("Testing Coefficient of Determination: " +
                          str(r2_score(self.y_test, pred_test_rr)))
                    # print("Total Time: " + str(EndTime - StartTime))

                    print("Order: " + str(i))

                    print("\nLasso: ")
<<<<<<< HEAD
                    model_lasso = Lasso(alpha=1)
                    StartTime = t.time()
=======
                    model_lasso = Lasso(
                        alpha=0.0001, tol=1e-1)

                    StartLTime = t.time()
>>>>>>> 66cd9b193b440e4b4bd9631d213cc0f9dd3b51ae
                    model_lasso.fit(self.X_train, self.y_train)
                    EndLTime = t.time()

                    pred_train_lasso = model_lasso.predict(self.X_train)

                    print(f"Lasso Training Time: {EndLTime - StartLTime:.5f}")
                    print("Intercept: " + str(model_lasso.intercept_))
                    print("Coefficients: " + str(model_lasso.coef_))
                    print(
                        "Training RMSE: " + str(np.sqrt(mean_squared_error(self.y_train, pred_train_lasso))))
                    print("Training Coefficient of Determination: " +
                          str(r2_score(self.y_train, pred_train_lasso)))

                    pred_test_lasso = model_lasso.predict(self.X_test)
                    print(
                        "Testing RMSE: " + str(np.sqrt(mean_squared_error(self.y_test, pred_test_lasso))))
                    print("Testing Coefficient of Determination: " +
                          str(r2_score(self.y_test, pred_test_lasso)))

                    # print("Total Time: " + str(EndTime - StartTime))

                    print("Order: " + str(i))

                    # Elastic Net
                    print("\nElastic Net: ")
<<<<<<< HEAD
                    model_enet = ElasticNet(alpha=1)
                    StartTime = t.time()
=======
                    model_enet = ElasticNet(
                        alpha=0.0001, tol=1e-1)

                    StartEnTime = t.time()
>>>>>>> 66cd9b193b440e4b4bd9631d213cc0f9dd3b51ae
                    model_enet.fit(self.X_train, self.y_train)
                    EndEnTime = t.time()

                    pred_train_enet = model_enet.predict(self.X_train)

                    print(
                        f"Elastic Net Training Time: {EndEnTime - StartEnTime:.5f}")
                    print("Intercept: " + str(model_enet.intercept_))
                    print("Coefficients: " + str(model_enet.coef_))
                    print(
                        "Training RMSE: " + str(np.sqrt(mean_squared_error(self.y_train, pred_train_enet))))
                    print("Training Coefficient of Determination: " +
                          str(r2_score(self.y_train, pred_train_enet)))

                    pred_test_enet = model_enet.predict(self.X_test)
                    print(
                        "Testing RMSE: " + str(np.sqrt(mean_squared_error(self.y_test, pred_test_enet))))
                    print("Testing Coefficient of Determination: " +
                          str(r2_score(self.y_test, pred_test_enet)))

                    #print("Total Time: " + str(EndTime - StartTime))

                    print("\n\n")


if __name__ == "__main__":

    # Initialize the object
<<<<<<< HEAD
    lre = LassoRidgeElastic("Data1.csv", 6, 4)
=======
    lre = LassoRidgeElastic("Data1.csv", 8, 4)
>>>>>>> 66cd9b193b440e4b4bd9631d213cc0f9dd3b51ae
    #gradient_descent = gd("Data1.csv")

    myProblems = [Question.NormReg]

    for Problem in myProblems:

        match Problem:

            case Question.NormReg:

                lre.train(Problem)
                #print("Lin Regr")

            case Question.K_Fold:

                lre.train(Problem)
                #print("Feature Scaling")