import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta       
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv('ENB2012_data.csv')
df

#note to self :

#input parameters
# X1 Relative Compactness
# X2 Surface Area
# X3 Wall Area
# X4 Roof Area
# X5 Overall Height
# X6 Orientation
# X7 Glazing Area
# X8 Glazing Area Distribution

# outputs
# y1 Heating Load
# y2 Cooling Load


def check_clean_data(df):
    print(df.isnull().sum()) #check for null values
    print("\n(Rows, Columns)", df.shape) #check for missing values

    print("\nCheck for empty values")
    print(df[df.eq("?").any(1)]) #check for missing values
check_clean_data(df)

def standardize_data(df):
    feature_cols = [col for col in df.columns if col not in ['Y1', 'Y2']]
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean())/df[col].std()
    return df
df = standardize_data(df)


def df_stats(df):
    print(df.describe().round(6))


corr_df = df.corr().round(6)

#code taken from https://towardsdatascience.com/why-feature-correlation-matters-a-lot-847e8ba439c4#:~:text=Positive%20Correlation%3A%20means%20that%20if,they%20have%20a%20linear%20relationship.

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.3f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()

features = df.copy()
features = features.drop(columns=["Y1", "Y2"])


labels = df.copy()
labels = labels.drop(columns=["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"])

features = features.to_numpy()
labels = labels.to_numpy()


class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass
    
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]                         #add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([np.ones(N),x])    #add bias by adding a constant feature of value 1
        self.w = np.linalg.inv(x.T @ x)@x.T@y  
        print("w", self.w)
        # self.w = np.linalg.lstsq(x, y)[0]  #  #return w for the least square difference
        # print("lstsq", self.w)
        return self
    
    def predict(self, x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([np.ones(N),x]) #change order
        yh = x@self.w                             #predict the y values
        return yh


X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2, random_state=13)


model = LinearRegression()
model.fit(X_train, y_train)
yh = model.predict(X_test)

print("yh", np.array(yh).shape)
print("y_test", np.array(y_test).shape)


def mean_squared_error(matrix1, matrix2):
    # Calculate the difference between the matrices
    difference = matrix1 - matrix2
    # Calculate the squared difference
    squared_difference = difference**2
    # Calculate the mean of the squared difference
    mean_squared_difference = np.mean(squared_difference)
    # Return the mean squared error
    return mean_squared_difference

def mean_absolute_error(matrix1, matrix2):
    # Calculate the absolute difference between the matrices
    difference = np.abs(matrix1 - matrix2)
    # Calculate the mean of the absolute difference
    mean_absolute_difference = np.mean(difference)
    # Return the mean absolute error
    return mean_absolute_difference

def r_squared(matrix1, matrix2):
    # Calculate the mean of the first matrix
    mean1 = np.mean(matrix1)
    # Calculate the difference between the matrices and the mean of the first matrix
    difference = matrix1 - matrix2
    mean_difference = matrix1 - mean1
    # Calculate the sum of squared differences
    squared_difference = difference**2
    mean_squared_difference = mean_difference**2
    # Calculate the R-squared
    r_squared = 1 - (np.sum(squared_difference) / np.sum(mean_squared_difference))
    # Return the R-squared
    return r_squared


print("MSE", mean_squared_error(y_test, yh))
print("MAE", mean_absolute_error(y_test, yh))
print("R^2", r_squared(y_test, yh))

def apply_linear_regression(test_size,features,labels):
  X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=test_size, random_state=54)
  
  model = LinearRegression()
  model.fit(X_train, y_train)
  yh = model.predict(X_test)

  error = mean_squared_error(y_test,yh)

  return error

sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
errors = []
for i in range(1,10):
  calc_error = apply_linear_regression((i/10),features,labels)
  errors.append(calc_error)
print(errors)

