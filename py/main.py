from sklearn.datasets import fetch_openml

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Extract features and labels
X, y = mnist["data"], mnist["target"]

# Optionally, convert y to integer type
y = y.astype(int)



# Get the first sample and reshape it to 28x28
first_sample = X.iloc[0].to_numpy().reshape(28, 28)

# Define the PPM header
header = f"P3\n28 28\n255\n"

# Create the PPM data
ppm_data = header
for row in first_sample:
    for pixel in row:
        # Each pixel needs 3 values (RGB), we replicate the grayscale value
        ppm_data += f"{int(pixel)} {int(pixel)} {int(pixel)} "

    ppm_data += "\n"

# Write the PPM data to a file
with open("tmp_first_sample.ppm", "w") as f:
    f.write(ppm_data)
print(f'first sample label: {y[0]}')
print("PPM file 'first_sample.ppm' created successfully.")
print("\n")




# Normalize the data to keep gradients manageble
X = X / 255

import numpy as np
import pandas as pd

# Overwrite the labels (y) so that all zeros have label 1 and everything else has lable 0
y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new

# Reshape and split the dataset into a training and test part
m = 60000 # number of samples in the training set 
m_test = X.shape[0] - m # X.shape[0] returns total amount of samples in X -> test_m is 10000

X_train = X[:m].T                 # Slice and transpose
X_test  = X[m:].T                 # Slice and transpose
y_train = y[:m].reshape(1,m)      # slice and make into a matrix
y_test  = y[m:].reshape(1,m_test) # slice and make into a matrix

# Shuffle the training data
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train = X_train.iloc[:, shuffle_index]
y_train = y_train[:, shuffle_index]

# Select a sample at random and plot it, to check that it looks ok
# import matplotlib
# import matplotlib.pyplot as plt

# i = 3
# plt.imshow(X_train.iloc[:,i].to_numpy().reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()
# print(y_train[:,i])


# Define sigmoid 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define cost function
def compute_loss(Y: np.ndarray, Y_hat: pd.core.frame.DataFrame) -> float:
    Y_hat = np.array(Y_hat) # Convert Y_hat to a NumPy array

    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )
    return L



# Build and train network
learning_rate = 1

X = X_train 
Y = y_train

n_x = X.shape[0] # n_x = number of rows in X
m = X.shape[1]   # m   = number of cols in X
'''
    structure of X:
    x_i,j = x_pixelidx,sampleidx
    -> every column is the pixels of each image in the set

    x_0,0   x_0,1   x_0,2   ... x_0,59999
    x_1,0   x_1,1   x_1,2   ... x_1,59999
    .       .
    .               .
    .                       .
    x_784,0 x_784,1 x_784,2 ... x_784,59999
'''

# Initialize W and b
W = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))

# Perform the training and plot cost progress
number_of_epochs = 2000
for i in range(number_of_epochs):
    Z = np.matmul(W.T, X) + b
    A = sigmoid(Z)

    cost = compute_loss(Y, A)

    dW = (1/m) * np.matmul(X, (A-Y).T)
    # db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)
    db = (1/m) * np.sum(A-Y, axis=1).to_numpy().reshape(b.shape)


    W = W - learning_rate * dW
    b = b - learning_rate * db

    if (i % 100 == 0):
        print("Epoch", i, "cost:\t\t", cost)

print("Final cost:\t", cost)



# Examine the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

Z = np.matmul(W.T, X_test) + b
A = np.array(sigmoid(Z)) # convert from pandas.core.frame.DataFrame to numpy.ndarray
print("y_test.shape[0]:", y_test.shape[0])
print("y_test.shape[1]:", y_test.shape[1])

predictions = (A>.5)[0,:]   # Replace all elements in [[A0, A1, ... , A9999]] with Ai>0.5, and extract the inner array. Example: [0, 0, 1, ..., 0, 1]
labels = (y_test == 1)[0,:] # Replace all elements in [[y_test0, y_test1, ... , y_test9999]] with y_testi>0.5, and extract the inner array. Example: [0, 0, 1, ..., 0, 1]

print(confusion_matrix(predictions, labels))
'''
    TP FN
    FP TN
What we want: 
    high low
    low  high
'''

print("\n")
print(classification_report(predictions, labels))

print("\nDone! :)")