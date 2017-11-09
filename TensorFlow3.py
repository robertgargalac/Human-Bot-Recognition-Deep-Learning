import tensorflow as tf
from pandas import*
from csv import DictReader
import random 
import numpy

# Function which create shuffled batches given a list of shuffled indexes and the path of file
def create_batch(index_batch, path):
    feature_list = []
    input_file = open(path, 'r')
    line = input_file.readlines()
    for index in index_batch:
        feature_list.append(line[index])

    X = numpy.asarray(feature_list)
    return X

def split_data(data, chunk_size):
    listofdf = list()
    number_chunks = len(data) // chunk_size + 1
    for i in range(number_chunks):
        listofdf.append(data[i*chunk_size:(i+1)*chunk_size])
    return listofdf

count = 0
target_list_train = []
target_list_validation = []
number_of_lines = 0
chunk_size = 35
epochs = 1
# Defining Placeholders (features and labels):

X = tf.placeholder(tf.float32, [None, 600])
Y_ = tf.placeholder(tf.float32, [None, 1])

# Defining Weights and Biases ( 2 hidden layers)

W1 = tf.Variable(tf.truncated_normal([600, 1024], stddev=0.1))
B1 = tf.Variable(tf.ones([1024])/10)
W2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
B2 = tf.Variable(tf.ones([512])/10)
W3 = tf.Variable(tf.truncated_normal([512, 1], stddev=0.1))
B3 = tf.Variable(tf.zeros([1]))

# Model
def neural_net(X):
    Y1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
    Y2 = tf.nn.relu(tf.add(tf.matmul(Y1, W2), B2))
    Y = tf.matmul(Y2, W3) + B3
    return Y

# Construct Model
logits = neural_net(X)
prediction = tf.sigmoid(logits)

# Loss function and optimizer:
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_, logits=logits)
cross_entropy = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
train_op = optimizer.minimize(cross_entropy)

#Performance Parameters

predicted_class = tf.greater(prediction, 0.5)
correct = tf.equal(predicted_class, tf.equal(Y_, 1.0))
acc = tf.reduce_mean(tf.cast(correct, 'float'))

# Get file lines number
with open('traintf.csv') as file:
    read = DictReader(file, delimiter=',')
    for row in read:
        number_of_lines += 1

print('Number of lines', number_of_lines)

# Get target values for train
with open('train.txt') as file1:
    reader = DictReader(file1, delimiter='\t')
    for row in reader :
        target_list_train.append(int(row['human-generated']))
y_train = numpy.asarray(target_list_train)
y_train = y_train[:, numpy.newaxis]

#Read validation file
reader_validation = pandas.read_csv('validationtf.csv', iterator=True, chunksize=1, delimiter=',')

#Get target values for validation
with open('validation.txt')as file2:
    reader = DictReader(file2, delimiter='\t')
    for row in reader:
        target_list_validation.append(int(row['human-generated']))
y_validation = numpy.asarray(target_list_validation)
y_validation = y_validation[:, numpy.newaxis]

index_list = range(number_of_lines)


    # Train the model
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
print('The session is starting')
with tf.Session() as sess:
    sess.run(init)
    for _ in range(epochs):

        print('Epoch' + str(_) + 'is executing')
        shuffled_index_list = random.sample(index_list, len(index_list))
        batch_shuffle = split_data(shuffled_index_list, chunk_size)
        y_train_shuffled = [y_train[index] for index in shuffled_index_list]
        y_train_batch = split_data(y_train_shuffled, chunk_size)

        for k in range(0, int(number_of_lines / chunk_size)):
            print('Number of iterations:', k)
            batch_X, batch_Y = create_batch(batch_shuffle[k], 'traintf.csv'), y_train_batch[k]
            print('Check1')
            train_data = {X: batch_X, Y_: batch_Y}
            print('Check2')
            sess.run(train_op, feed_dict=train_data)
            print('Check3')

            # Calculate batch loss and accuracy
            loss, score = sess.run([cross_entropy, acc], feed_dict=train_data)
            w1, w2, w3 = sess.run([W1, W2, W3], feed_dict=train_data)
            logits_val = sess.run([logits], feed_dict=train_data)
            print("Step " + str(k))
            print("Minibatch Loss=", loss)
            print("Training Accuracy=", score)
            print("First weight", w1)
            print("Second weight", w2)
            print("Third weight", w3)


        print("The train phase is done")

    for chunk_val in reader_validation:

        chunk_val = numpy.nan_to_num(chunk_val)
        validation_data = {X: chunk_val, Y_: y_validation}
        count += 1
        print('VALIDATION PHASE:')
        print("Predictions:", sess.run(prediction, feed_dict=validation_data))







