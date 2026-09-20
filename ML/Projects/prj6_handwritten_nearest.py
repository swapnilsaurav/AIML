import numpy as np
import matplotlib.pyplot as plt
import time

#Load the training set:
train_data = np.load('data/train_data.npy')
train_labels = np.load('data/train_labels.npy')

#Load the testing set:
test_data = np.load('data/test_data.npy')
test_labels = np.load('data/test_labels.npy')

#Print out data dimensions:
print("Training dataset dimensions: ", np.shape(train_data))
print("Number of training labels: ", len(train_labels))
print("Testing dataset dimensions: ", np.shape(test_data))
print("Number of testing labels: ", len(test_labels))

#Compute the number of examples of each digit:
train_digits, train_counts = np.unique(train_labels, return_counts=True)
print("Training set distribution:")
print(dict(zip(train_digits, train_counts)))
test_digits, test_counts = np.unique(test_labels, return_counts=True)
print("Test set distribution:")
print(dict(zip(test_digits, test_counts)))

####################
#2. Visualizing the data
#Each data point is stored as 784-dimensional vector. To visualize a data point,
# we first reshape it to a 28x28 image.
#Define a function that displays a digit given its vector representation:
def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return

#Define a function that takes an index into a particular data set ("train" or "test")
# and displays that image.
def vis_image(index, dataset="train"):
    if(dataset=="train"):
        show_digit(train_data[index,])
        label = train_labels[index]
    else:
        show_digit(test_data[index,])
        label = test_labels[index]
    print("Actual Label " + str(label))
    return
#View the first data point in the training set:
vis_image(0, "train")
#View the first data point in the test set:
vis_image(0, "test")

################

#Function that computes squared Euclidean distance between two vectors.
def squared_dist(x,y):
    return np.sum(np.square(x-y))
#Compute distance between a seven and a one in our training set.
vis_image(4, "train")
vis_image(5, "train")
print("Distance from 7 to 1: ", squared_dist(train_data[4,],train_data[5,]))

#Compute distance between a seven and a two in our training set.
vis_image(4, "train")
vis_image(1, "train")
print("Distance from 7 to 2: ", squared_dist(train_data[4,],train_data[1,]))

#Compute distance between two seven's in our training set.
vis_image(4, "train")
vis_image(7, "train")
print("Distance from 7 to 7: ", squared_dist(train_data[4,],train_data[7,]))

#Compute distance between Same two seven's in our training set.
vis_image(4, "train")
vis_image(4, "train")
print("Distance from 7 to Same 7: ", squared_dist(train_data[4,],train_data[4,]))
###################
#4. Computing nearest neighbors
#Now that we have a distance function defined,
# we can now turn to nearest neighbor classification.

#Function that takes a vector x and returns the index
# of its nearest neighbor in train_data.
def find_NN(x):
    # Compute distances from x to every row in train_data
    distances = [squared_dist(x,train_data[i,]) for i in range(len(train_labels))]
    # Get the index of the smallest distance
    return np.argmin(distances)

#Function that takes a vector x and returns the class of its nearest neighbor in train_data.
def NN_classifier(x):
    # Get the index of the the nearest neighbor
    index = find_NN(x)
    # Return its class
    return train_labels[index]
#A success case:
print("A success case:")
print("NN classification: ", NN_classifier(test_data[0,]))
print("True label: ", test_labels[0])
print("The test image:")
vis_image(0, "test")
print("The corresponding nearest neighbor image:")
vis_image(find_NN(test_data[0,]), "train")

#A failure case:
print("A failure case:")
print("NN classification: ", NN_classifier(test_data[39,]))
print("True label: ", test_labels[39])
print("The test image:")
vis_image(39, "test")
print("The corresponding nearest neighbor image:")
vis_image(find_NN(test_data[39,]), "train")

#######
#5. Processing the full test set
t_before = time.time()
test_predictions = [NN_classifier(test_data[i,]) for i in range(len(test_labels))]
t_after = time.time()
#Compute the error:
err_positions = np.not_equal(test_predictions, test_labels)
error = float(np.sum(err_positions))/len(test_labels)
print("Error of nearest neighbor classifier: ", error)
print("Classification time (seconds): ", t_after - t_before)

########
#6. Faster nearest neighbor methods
from sklearn.neighbors import BallTree
# Build nearest neighbor structure on training data
t_before = time.time()
ball_tree = BallTree(train_data)
t_after = time.time()
# Compute training time
t_training = t_after - t_before
print("Time to build data structure (seconds): ", t_training)
# Get nearest neighbor predictions on testing data
t_before = time.time()
test_neighbors = np.squeeze(ball_tree.query(test_data, k=1, return_distance=False))
ball_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
# Compute testing time
t_testing = t_after - t_before
print("Time to classify test set (seconds): ", t_testing)
# Verify that the predictions are the same
print("Ball tree produces same predictions as above? ", np.array_equal(test_predictions, ball_tree_predictions))

from sklearn.neighbors import KDTree
# Build nearest neighbor structure on training data
t_before = time.time()
kd_tree = KDTree(train_data)
t_after = time.time()
# Compute training time
t_training = t_after - t_before
print("Time to build data structure (seconds): ", t_training)
# Get nearest neighbor predictions on testing data
t_before = time.time()
test_neighbors = np.squeeze(kd_tree.query(test_data, k=1, return_distance=False))
kd_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
# Compute testing time
t_testing = t_after - t_before
print("Time to classify test set (seconds): ", t_testing)
# Verify that the predictions are the same
print("KD tree produces same predictions as above? ", np.array_equal(test_predictions, kd_tree_predictions))
