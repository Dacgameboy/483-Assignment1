import pandas as pd

# Can be adjusted as needed
alpha = 0.1

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


# w - A list of each feature's current value (the thing we're training)
# x - A list containing every point we're plugging into the summation
# y - A list containing the expected output value
# alpha - The learning rate
# num_examples - The number of rows in x
def train_data(w, x, y, alpha, num_examples):


df = pd.read_csv("Data1.csv")

N = len(df)

order = 1

# Original features
orig_col_names = list(df.columns[:-1])


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


