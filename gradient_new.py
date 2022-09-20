import pandas as pd
import numpy as np

# Can use numpy for matrix multiplication

def Residual_Sum(w, x, y, num_examples, getMean = False):
    sum = 0

    if not getMean:
        for i in range(num_examples):
            predict_val = w[0]
            predict_val += np.dot(w[1:], x[i])
            sum += (y[i, 0] - predict_val)**2
        return sum

    else:
        mean = 0
        for i in range(num_examples):
            predict_y = w[0]
            predict_y += np.dot(w[1:], x[i])
            sum += (y[i, 0] - predict_y)**2
            mean += predict_y
        mean /= num_examples

    return sum, mean

def RMSE(w, x, y, num_examples):
    RMSE = (Residual_Sum(w, x, y, num_examples)/num_examples) ** (1/2)
    return RMSE

def R_Sq(w, x, y, num_examples):
    RSS, Mean = Residual_Sum(w, x, y, num_examples, True)
    sum = 0
    for i in range(num_examples):
       sum += (y[i, 0] - Mean) ** 2
    
    return 1 - (RSS/sum)
    

# Can be adjusted as needed
alpha = 0.008 #0.008

classifier = "Y"

# This function adds a new column for each ORIGINAL column that existed in the dataset
# with raised to a power of "order"
#
# df - Data frame
# data_len - Number of examples (N)
# orig_col_names - A list of the original feature names
# order - The order to increase to
def inc_order(w, training_x, testing_x, orig_col_names, order):

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

        #training_x = pd.concat([training_x, training_col_data], axis = 1)

        #testing_x = pd.concat([testing_x, testing_col_data], axis = 1)

        training_x[name] = training_col_data

        #training_x.insert(0, name, training_col_data, True)

        testing_x[name] = testing_col_data

        #testing_x.insert(0, name, testing_col_data, True)

        w = np.append(w, [1])

        return w

# Dot Product
#
# w - An array of w values
# x - An array containing one example
def get_predict_val(w, x):

    x.reset_index(drop=True, inplace=True)

    #if len(w) != len(x.columns) + 1:

    #    print(len(w))
    #    print(len(x.columns))
    #    exit(1)

    # The sum starts with the intercept
    predict_val = w[0]

    #print("x's length: " + str(len(x.columns)))

    for i in range(len(x.columns)):

        predict_val += (w[i + 1] * x[0, i])

        #print("Multiplying " + str(w[i + 1]) + " and " + str(x.iat[0, i]))

    return predict_val


# w - A list of each feature's current value (the thing we're training)
# x - A list containing every point we're plugging into the summation
# y - A list containing the expected output value
# alpha - The learning rate
# num_examples - The number of rows in x
def train_data(w, x, y, alpha, num_examples):

    N = len(w)

    #v = []
    v = np.array([])
    
    for j in range(N):

        print("Training w at index " + str(j) + "...")

        sum = 0

        for i in range(num_examples):

            if j == 0:

                x_val = 1

            else:

                x_val = x[i, j - 1]

            #print("Adding (" + a + " - " + b + ") * " + c + " to the sum...")

            predict_val = w[0]

            predict_val += np.dot(w[1:], x[i])

            sum += (predict_val - y[i, 0]) * x_val

            #print(sum)

        #print("The gradient is: " + str(sum))

        sum *= (alpha / num_examples)


        #print("v before: " + str(v))
        v = np.append(v, [w[j] - sum])
        #print("v after: " + str(v))

        #w[j] -= sum

    #print("V = " + str(v))

    return v
            

# Transform pandas df to numpy array
#df = pd.read_csv("Data1.csv")
#df = pd.read_csv("Test_Data.csv")
df = pd.read_csv("Test_Data_2.csv")

N = len(df)

# Original features
orig_col_names = list(df.columns[:-1])

# initial values for w
#w = [0.01] * (len(orig_col_names) + 1)

w = np.array([1] * (len(orig_col_names) + 1))

# Take first sample (training)
#df_new = df.sample(frac = 0.75, replace=True)

# Remove classifier
training_x = df.loc[:, df.columns != classifier]

# Remove features
training_y = df.loc[:, df.columns == classifier]

# Take second sample (testing)
df_new = df.sample(frac = 0.25, replace=True)

# Remove classifier
testing_x = df_new.loc[:, df.columns != classifier]

# Remove features
testing_y = df_new.loc[:, df.columns == classifier]


training_x.reset_index(drop=True, inplace=True)
testing_x.reset_index(drop=True, inplace=True)

training_y.reset_index(drop=True, inplace=True)
testing_y.reset_index(drop=True, inplace=True)

#inc_order(w, training_x, testing_x, orig_col_names, 1)
w = inc_order(w, training_x, testing_x, orig_col_names, 2)
#inc_order(w, training_x, testing_x, orig_col_names, 3)

#for i in range(2, 10):

#    inc_order(w, training_x, testing_x, orig_col_names, i)


print(training_x)
print(training_y)

print("w: " + str(w))

for i in range(1500):

    # Convert training_x and training_y to numpy
    training_x_numpy = training_x.to_numpy()    
    training_y_numpy = training_y.to_numpy()

    #print(training_x_numpy[1])

    w = train_data(w, training_x_numpy, training_y_numpy, alpha, len(training_x_numpy))

    print("w: " + str(w))

    print("RMSE: " + str(RMSE(w, training_x_numpy, training_y_numpy, len(training_x_numpy))))

    print("R^2: " + str(R_Sq(w, training_x_numpy, training_y_numpy, len(training_x_numpy))))