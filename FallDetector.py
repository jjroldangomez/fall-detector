import tensorflow as tf
import numpy as np
import csv
import copy
import time
import os

import matplotlib.pyplot as plt

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# Create a dataset with falls to train the model:
def create_fall_dataset(vars, win):

    # Initialize datasets:
    train_X = [] # Training dataset: Measurements of accelerometer and/or gyroscope.
    train_y = [] # Training dataset: Stand or fall?
    test_X = [] # Test dataset: Measurements of accelerometer and/or gyroscope.
    test_y = [] # Test dataset: Stand or fall?

    # For each data file...
    for n in range(1, 5): # I'm taking 5 instead of 20 to save time.

        # Get the path of data file:

        dir = os.path.dirname(__file__)
        path = os.path.join(dir, 'Simulated/' + str(n) + '.csv')

        # Open the file:
        csvfile = open(path)

        # Read the data:
        reader = csv.reader(csvfile, delimiter=',')

        # Initialize auxiliary variables:

        all_x = [] # Buffer for all inputs.
        all_y = [] # Buffer for all outputs.

        buffer_x = [] # Buffer for one input.
        for j in range(len(vars)):
            buffer_x.append([])

        index = 0 # Counter for rows.
        row_count = len(list(reader)) # Total number of rows.

        save = 0 # Flag to save data.

        # Open the file again:
        csvfile = open(path)

        # Read the data again:
        reader = csv.reader(csvfile, delimiter=',')

        # For each row...
        for row in reader:

            buffer_y = [] # Buffer for one output.
            buffer_y.append(0) # y(1)=1 Stand
            buffer_y.append(0) # y(2)=1 Fall

            if index > 2 and index < row_count-1:

                # Read output:
                if float(row[65]) < 50:
                    buffer_y[0] = 1
                else:
                    buffer_y[1] = 1

                # Read input:

                # For each input...
                for j in range(len(vars)):

                    # Add variable to buffer:
                    buffer_x[j].append(float(row[vars[j]]))

                    # If the buffer is complete...
                    if len(buffer_x[j]) > win:

                        aux = buffer_x[j][1:win+1]
                        buffer_x[j] = aux

                        save = 1

                if save > 0:

                    aux=[]

                    for k in range(len(buffer_x[j])):
                        for j in range(len(vars)):
                            aux.append(buffer_x[j][k])

                    all_x.append(copy.deepcopy(aux))
                    all_y.append(copy.deepcopy(buffer_y))

            index += 1

        # Split data into train and test datasets:
        for i in range(0, len(all_x)):
            if i % 10 == 0: # I set train 90% and test 10%. You can change it!
                test_X.append(copy.deepcopy(all_x[i]))
                test_y.append(copy.deepcopy(all_y[i]))
            else:
                train_X.append(copy.deepcopy(all_x[i]))
                train_y.append(copy.deepcopy(all_y[i]))

    return train_X, test_X, train_y, test_y

# Load data of one fall to test the model:
def create_fall_data(vars, win, sample):

    # Get the path of data file:
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, sample)

    # Open the file:
    csvfile = open(path)

    # Read the data:
    reader = csv.reader(csvfile, delimiter=',')

    # Initialize variables:

    all_x = [] # Buffer for all inputs.
    all_y = [] # Buffer for all outputs.

    buffer_x = [] # Buffer for one input.
    for j in range(len(vars)):
        buffer_x.append([])

    index = 0 # Counter for rows.
    row_count = len(list(reader)) # Total number of rows.

    save = 0 # Flag to save data.

    # Open the file again:
    csvfile = open(path)

    # Read the data again:
    reader = csv.reader(csvfile, delimiter=',')

    # For each row...
    for row in reader:

        buffer_y = [] # Buffer for one output.
        buffer_y.append(0) # y(1)=1 Stand
        buffer_y.append(0) # y(2)=1 Fall

        if index > 2 and index < row_count - 1:

            # Read output:
            if float(row[65]) < 50:
                buffer_y[0] = 1
            else:
                buffer_y[1] = 1

            # Read input:

            # For each input...
            for j in range(len(vars)):

                # Add variable to buffer:
                buffer_x[j].append(float(row[vars[j]]))

                # If the buffer is complete...
                if len(buffer_x[j]) > win:
                    aux = buffer_x[j][1:win + 1]
                    buffer_x[j] = aux

                    save = 1

            if save > 0:

                aux = []

                for k in range(len(buffer_x[j])):
                    for j in range(len(vars)):
                        aux.append(buffer_x[j][k])

                all_x.append(copy.deepcopy(aux))
                all_y.append(copy.deepcopy(buffer_y))

        index += 1

    return all_x, all_y

# Main function to train the model:
def main_train(model):

    # DEFINE THE DATA FLOW GRAPH:
    # 1) Architecture of network.
    # 2) Process of forward propagation.
    # 3) Process of back propagation.

    # Create a dataset with falls:
    train_X, test_X, train_y, test_y = create_fall_dataset([52, 53, 54], 10) # I'm using three variables. You can try others!

    # TODO: Define the sizes of neural network
    # 1) x_size: Number of nodes in the input.
    # 2) h_size: Number of nodes in hidden layer.
    # 3) y_size: Number of nodes in the output.
    # Hint: x_size and y_size are defined by the dataset. h_size can be changed to improve performance.
    x_size = 0
    h_size = 0
    y_size = 0

    # Create placeholders for inputs and outputs:
    X = tf.placeholder("float", shape=[None, x_size], name="X")
    y = tf.placeholder("float", shape=[None, y_size], name="Y")

    # TODO: Initialize weights and biases
    # Hints:
    # - Choose the correct sizes according to the inputs and outputs of every layer.
    # - Use the function tf.Variable to define the weights as Tensorflow variables.
    # - Use the function tf.random_normal to provide random values for weights and biases.
    w_1 = 0
    w_2 = 0
    b_1 = 0
    b_2 = 0

    # TODO: Forward propagation
    # Input layer: h = w1 * X + b1
    # Hidden layer: yhat = w2 * h + b2
    # Hints:
    # - Use tf.matmul for multiplying tensors.
    # - Use tf.nn.relu as activation function for input layer.
    # - Use tf.nn.softmax as activation function for hidden layer.
    # - Name the result of hidden layer as "Yhat" for latter computations.
    h = 0
    yhat = 0

    # Make prediction:
    predict = tf.argmax(yhat, axis=1)

    # TODO: Backward propagation
    # Compute the cost as the difference between the expected and computed outputs.
    # Hint: Use tf.reduce_mean and tf.nn.softmax_cross_entropy_with_logits.
    # Change the weights searching to minimize the cost.
    # Hint: Adam Optimizer with learning rate of 0.001 provides good results.
    cost = 0
    updates = 0

    # Compute error for visualization:
    mean_squared_error = tf.reduce_mean(tf.square(tf.subtract(yhat, y)))

    # APPLY THE DATA FLOW GRAPH:
    # 1) Load the data flow graph.
    # 2) Train the neural network.
    # 3) Save the resultant model.

    # Initialize Tensorflow session:
    sess = tf.Session()

    # Initialize global variables:
    init = tf.global_variables_initializer()

    # Load global variables in Tensorflow session:
    sess.run(init)

    # Initialize auxiliary variables:
    train_acc = [] # Accuracy with training dataset.
    test_acc = [] # Accuracy with test dataset.

    num_epochs = 100 # Number of epochs for optimization.

    # For each epoch...
    for epoch in range(num_epochs):

        # For each sample...
        for i in range(len(train_X)):

            # TODO: Run session to train neural network
            # Hints:
            # - You must use sess.run() with the adequate parameters.
            # - You want to compute "updates" and "mean_squared_errors".
            # - You need to provide "train_X[i]" and "train_y[i]".
            sess.run()

        # Compute train and test accuracies:
        train_accuracy = np.mean(np.argmax(np.asarray(train_y), axis=1) == sess.run(predict, feed_dict={X: np.asarray(train_X), y: np.asarray(train_y)}))
        test_accuracy = np.mean(np.argmax(np.asarray(test_y), axis=1) == sess.run(predict, feed_dict={X: np.asarray(test_X), y: np.asarray(test_y)}))

        # Add accuracies to list:
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        # Show accuracies in screen:
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    # Plot accuracies in graphic:
    plt.plot(np.squeeze(test_acc))
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.show()

    # Create a saver:
    saver = tf.train.Saver()

    # Save the graph:
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, model)
    saver.save(sess, path, global_step=num_epochs)

    # Close session:
    sess.close()

# Main function to make predictions:
def main_predict(model, sample):

    # Initialize global variables:
    init = tf.global_variables_initializer()

    # Initialize Tensorflow session:
    with tf.Session() as sess:

        # Load global variables in Tensorflow session:
        sess.run(init)

        # Load the saved model:
        dir = os.path.dirname(__file__)
        path = os.path.join(dir, model)

        saver = tf.train.import_meta_graph(path)

        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()

        # Get the saved tensors:
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        Y_pred = graph.get_tensor_by_name("Yhat:0")

        # Load data of one fall:
        x_data, y_data = create_fall_data([52, 53, 54], 10, sample)

        # Initialize auxiliary variables:
        predicted = [] # Save predicted values.
        actual = [] # Save actual values.

        # For each sample...
        for i in range(len(x_data)):

            # Get input and actual output:
            input = [x_data[i]]
            output = y_data[i]

            # TODO: Prediction with model
            # Hints:
            # - Use sess.run with the adequate parameters.
            # - You want to obtain the predicted output.
            # - You have to provide the input and a default output ([0, 0]).
            y_pred = sess.run()

            if y_pred[0][0] < y_pred[0][1]:
                y_pred[0][0] = 0
                y_pred[0][1] = 1
            else:
                y_pred[0][0] = 1
                y_pred[0][1] = 0

            # Print on screen:
            print("Predicted: " + str(y_pred) + " / Actual: " + str(output))

            time.sleep(0.01)

            # Save results:
            predicted.append(np.argmax(y_pred))
            actual.append(np.argmax(y_data[i]))


        # Filter of output:
        number = 10

        for i in range(1, len(predicted)-number):
            equal = 1
            for j in range(i, i+number):
                if predicted[i] != predicted[j]:
                    equal = -1
            if equal == -1:
                predicted[i] = predicted[i-1]

        predicted.reverse()
        predicted.append(0)
        predicted.append(0)
        predicted.append(0)
        predicted.append(0)
        predicted.append(0)
        predicted.reverse()

        # Plot predicted and actual values:
        plt.plot(np.squeeze(predicted))
        plt.plot(np.squeeze(actual))
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.show()

# Main:
if __name__ == '__main__':

    flag = True # True for train, False for predict.

    if flag == True:
        # Train model
        model = 'my_model'
        main_train(model)
    else:
        # Make prediction
        model = 'my_model-100.meta' # Choose the model
        sample = 'Simulated/6.csv' # Choose the dataset
        main_predict(model, sample)