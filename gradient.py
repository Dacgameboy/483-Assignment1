import pandas as pd
import numpy as np

np.set_printoptions(suppress = True)

# Can use numpy for matrix multiplication
class Gradient:

    def __init__(self, filename, alpha, classifier, order, iterations):

        self.iterations = iterations

        self.order = order

        self.alpha = alpha

        self.classifier = classifier

        self.read_file(filename)

        self.orig_col_names = list(self.df.columns[1:-1])

        self.create_w([1] * (len(self.orig_col_names) + 1))
        #self.create_w([1, 303.120603174602093, 121.04723619047618, 0.03469134254241905, 415.238296909286])

        self.sample_sets()

        if self.order > 1:

            self.inc_order()
        
    
    def Residual_Sum(self, x, y, getMean = False):
        sum = 0
        num_examples = len(x)

        if not getMean:
            
            for i in range(num_examples):
                #predict_val = w[0]
                predict_val = np.dot(self.w, x[i])
                sum += (y[i, 0] - predict_val)**2
            return sum

        else:
            mean = 0
            for i in range(num_examples):
                #predict_y = w[0]
                predict_y = np.dot(self.w, x[i])
                sum += (y[i, 0] - predict_y)**2
                mean += predict_y
            mean /= num_examples

        return sum, mean

    def RMSE(self, x, y):
        num_examples = len(x)
        RMSE = (self.Residual_Sum(x, y)/num_examples) ** (1/2)
        return RMSE

    def R_Sq(self, x, y):
        num_examples = len(x)
        RSS, Mean = self.Residual_Sum(x, y, True)
        sum = 0
        for i in range(num_examples):
            sum += (y[i, 0] - Mean) ** 2
        
        return 1 - (RSS/sum)

    #inc_fluct_rate = 0.5 # 1.2

    #dec_fluct_rate = 0.5 # 0.14

    # This function adds a new column for each ORIGINAL column that existed in the dataset
    # with raised to a power of "order"
    #
    # df - Data frame
    # data_len - Number of examples (N)
    # orig_col_names - A list of the original feature names
    # order - The order to increase to
    def inc_order(self):

        train_len = len(self.training_x_numpy)
        test_len = len(self.testing_x_numpy)

        for i in range(len(self.orig_col_names)):

            # The data for the new column
            training_col_data = []
            testing_col_data = []

            for example in range(train_len):

                #new_col_data.append(df[col_name][example] ** order)
                training_col_data.append(self.training_x_numpy[example][i + 1] ** self.order)

            for example in range(test_len):
            
                testing_col_data.append(self.testing_x_numpy[example][i + 1] ** self.order)

            # Create the name for the column
            #name = str(i) + " Order: " + str(self.order)

            # Insert the column
            #df.insert(0, name, new_col_data)

            #training_x = pd.concat([training_x, training_col_data], axis = 1)

            #testing_x = pd.concat([testing_x, testing_col_data], axis = 1)

            self.training_x_numpy = np.insert(self.training_x_numpy, len(self.training_x_numpy[0]), [training_col_data], axis=1)

            #training_x.insert(0, name, training_col_data, True)

            self.testing_x_numpy = np.insert(self.testing_x_numpy, len(self.testing_x_numpy[0]), [testing_col_data], axis=1)

            #testing_x.insert(0, name, testing_col_data, True)

            self.w = np.append(self.w, [1])


    # w - A list of each feature's current value (the thing we're training)
    # x - A list containing every point we're plugging into the summation
    # y - A list containing the expected output value
    # alpha - The learning rate
    # num_examples - The number of rows in x
    def train_data(self, x, y):

        N = len(self.w)

        num_examples = len(x)

        # This is the new gradient / old gradient summed
        #g_change = 0

        #new_g = 0
        #old_g = 0

        #v = []
        #self.v = np.array([])

        v = np.array([])
        
        for j in range(N):

            #print("Training w at index " + str(j) + "...")

            sum = 0

            for i in range(num_examples):

                #if j == 0:

                #    x_val = 1

                #else:

                #    x_val = x[i, j - 1]

                x_val = x[i][j]

                #print("Adding (" + a + " - " + b + ") * " + c + " to the sum...")

                #predict_val = w[0]

                #print("Dotting " + str(self.w) + " and " + str(x[i]))

                self.predict_val = np.dot(self.w, x[i])

                # Gradient
                sum += (self.predict_val - y[i][0]) * x_val
                
            print("Gradient for index " + str(j) + ": " + str(sum))

            #if gradient[j] != None:

                # Store the new and old gradient values
            #    new_g = abs(sum)
            #    old_g = abs(gradient[j])

                # Update the sum of all gradients, g_change
            #    if new_g > old_g:

            #        g_change += (new_g % old_g) / old_g

            #    elif new_g < old_g:

            #        g_change -= (new_g / old_g)
            
                #new_g += sum
                #old_g += gradient[j]

            # Update the old gradient
            #gradient[j] = sum

            # At this point, we have a gradient calculated

            #print("The gradient is: " + str(sum))

            sum = sum * (self.alpha / num_examples)

            v = np.append(v, [self.w[j] - sum])


            #v = np.append(self.v, [self.w[j] - sum])

            #w[j] -= sum

        # Check if we're overshooting on average
        #if g_change > 0:

        #    print("The gradient change is positive (" + str(g_change) + ")! Reducing alpha")
        #    alpha = (alpha * dec_fluct_rate) * (1 - (min(1, g_change) / N))

        #elif g_change < 0:

        #    print("The gradient change is negative(" + str(g_change) + ")! Increasing alpha")
        #    alpha = alpha * (1 + (((min(1, abs(g_change)) / N)) * inc_fluct_rate))

        #if old_g != 0:

        #    g_change += abs(new_g) / abs(old_g)

        #    alpha = (alpha * fluct_rate) * (1 - ((1 / N) * g_change))

        #return v #, alpha

        self.w = v
                

    def read_file(self, filename):

        self.df = pd.read_csv(filename)

        self.df.insert(0, "Constant", [1] * len(self.df))
        
        
    def create_w(self, arr):

        self.w = np.array(arr)
        
        
    def sample_sets(self):

        # Take first sample (training)
        df_new = self.df.sample(frac = 0.75, replace=True)

        # Remove classifier
        self.training_x_numpy = df_new.loc[:, self.df.columns != self.classifier]

        # Remove features
        self.training_y_numpy = df_new.loc[:, self.df.columns == self.classifier]

        # Take second sample (testing)
        df_new = self.df.sample(frac = 0.25, replace=True)

        # Remove classifier
        self.testing_x_numpy = df_new.loc[:, self.df.columns != self.classifier]

        # Remove features
        self.testing_y_numpy = df_new.loc[:, self.df.columns == self.classifier]


        self.training_x_numpy.reset_index(drop=True, inplace=True)
        self.testing_x_numpy.reset_index(drop=True, inplace=True)

        self.training_y_numpy.reset_index(drop=True, inplace=True)
        self.testing_y_numpy.reset_index(drop=True, inplace=True)

        self.training_x_numpy = self.training_x_numpy.to_numpy()    
        self.testing_x_numpy = self.testing_x_numpy.to_numpy()

        self.training_y_numpy = self.training_y_numpy.to_numpy()    
        self.testing_y_numpy = self.testing_y_numpy.to_numpy()

        print("Mean 1: " + str(self.training_x_numpy[:, 1].mean()))
        print("Mean 2: " + str(self.training_x_numpy[:, 2].mean()))
        print("Mean 3: " + str(self.training_x_numpy[:, 3].mean()))
        print("Mean 4: " + str(self.training_x_numpy[:, 4].mean()))

        #print(self.training_x_numpy)
        #print(self.training_y_numpy)

    # Original features
    

    # initial values for w
    #w = [1] * (len(orig_col_names) + 1)

    #w = np.array([0.01] * (len(orig_col_names) + 1))
    #w = np.array([0.999823,   -0.13085741,  0.01604778,  0.99986325,  0.21832722])

    #w = np.array([611.18305705,12485.69095535,-453.59327085,-1.74367274,-9023.79250752])

    #w = np.array([67.956433, 0.03722, -0.004231477, -5.58388, -0.0591527])

    # # Take first sample (training)
    # df_new = df.sample(frac = 0.02, replace=True)

    # # Remove classifier
    # training_x = df_new.loc[:, df.columns != classifier]

    # # Remove features
    # training_y = df_new.loc[:, df.columns == classifier]

    # # Take second sample (testing)
    # df_new = df.sample(frac = 0.25, replace=True)

    # # Remove classifier
    # testing_x = df_new.loc[:, df.columns != classifier]

    # # Remove features
    # testing_y = df_new.loc[:, df.columns == classifier]


    # training_x.reset_index(drop=True, inplace=True)
    # testing_x.reset_index(drop=True, inplace=True)

    # training_y.reset_index(drop=True, inplace=True)
    # testing_y.reset_index(drop=True, inplace=True)

    #inc_order(w, training_x, testing_x, orig_col_names, 1)
    #w = inc_order(w, training_x, testing_x, orig_col_names, 2)
    #w = inc_order(w, training_x, testing_x, orig_col_names, 3)
    #w = inc_order(w, training_x, testing_x, orig_col_names, 4)
    #w = inc_order(w, training_x, testing_x, orig_col_names, 5)


    #for i in range(2, 10):

    #    inc_order(w, training_x, testing_x, orig_col_names, i)


    # print(training_x)
    # print(training_y)

    # Convert training_x and training_y to numpy
    # training_x_numpy = training_x.to_numpy()    
    # training_y_numpy = training_y.to_numpy()

    #w = training_x_numpy[0]
    def train(self):

        print("w: " + str(self.w))

        print("Alpha: " + str(self.alpha))

        #gradient = [None] * len(w)

        for i in range(self.iterations):

            # Convert training_x and training_y to numpy
            #training_x_numpy = training_x.to_numpy()    
            #training_y_numpy = training_y.to_numpy()

            self.train_data(self.training_x_numpy, self.training_y_numpy)

            print("\n\n")

            print("w: " + str(self.w))

            print("RMSE: " + str(self.RMSE(self.training_x_numpy, self.training_y_numpy)))

            print("R^2: " + str(self.R_Sq(self.training_x_numpy, self.training_y_numpy)))

            #print("Alpha: " + str(alpha))

            #print("Gradient: " + str(self.gradient))




if __name__ == "__main__":

    gd = Gradient("Data1.csv", 0.000007, "Idx", 1, 20)

    gd.train()