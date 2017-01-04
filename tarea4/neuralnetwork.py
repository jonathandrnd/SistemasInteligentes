from __future__ import print_function
import tensorflow as tf
import numpy as np

inputs = 6
num_labels = 4
batch_size = 128
hidden1_units=1024
num_steps = 2001


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 7))
    # Map 0 to [1,0,0,...], 1 to [0,1,0,...]
    labels = (np.arange(num_labels) == labels[:, None])
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 *
            np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) /
            predictions.shape[0])


filenametrainset = "car.data"
filenametestset = "car-prueba.data"

# setup text reader
file_length_train = file_len(filenametrainset)
file_length_test = file_len(filenametestset)

filename_queue_train = tf.train.string_input_producer([filenametrainset])
filename_queue_test = tf.train.string_input_producer([filenametestset])

reader_train = tf.TextLineReader(skip_header_lines=0)
reader_test = tf.TextLineReader(skip_header_lines=0)

_, csv_row_train = reader_train.read(filename_queue_train)
__, csv_row_test = reader_test.read(filename_queue_test)


# setup CSV decoding
record_defaults = [['0'],['0'],['0'],['0'],['0'],['0'],['0']]
record_defaults2 = [['0'],['0'],['0'],['0'],['0'],['0']]

col1,col2,col3,col4,col5,col6,col7 = tf.decode_csv(csv_row_train, record_defaults=record_defaults)
colt1,colt2,colt3,colt4,colt5,colt6= tf.decode_csv(csv_row_test, record_defaults=record_defaults2)


# turn features back into a tensor
train_dataset = tf.pack([col1,col2,col3,col4,col5,col6])
train_labels = tf.pack([col7])
test_dataset = tf.pack([colt1,colt2,colt3,colt4,colt5,colt6])

print (file_length_train)
print (file_length_test)

ldataset=[]

valuesTrain = [[0 for x in range(inputs)] for y in range(file_length_train)]
valueslabel = [[0 for x in range(num_labels)] for y in range(file_length_train)]
valuesTest = [[0 for x in range(inputs)] for y in range(file_length_test)]

dict1= dict();
dict2= dict();
dict3= dict();
dict4= dict();
dict5= dict();
dict6= dict();
dict7= dict();

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    cont1=0.0;cont2=0.0;cont3=0.0;cont4=0.0;cont5=0.0;cont6=0.0;cont7=0.0;
    for i in range(file_length_train):
        a,b = sess.run([train_dataset, train_labels])
        if(dict1.get(a[0],-1)==-1):
            dict1[a[0]]=cont1;
            cont1=cont1+1;
        if (dict2.get(a[1], -1) == -1):
            dict2[a[1]] = cont2;
            cont2 = cont2 + 1;
        if (dict3.get(a[2], -1) == -1):
            dict3[a[2]] = cont3;
            cont3 = cont3 + 1;
        if (dict4.get(a[3], -1) == -1):
            dict4[a[3]] = cont4;
            cont4 = cont4 + 1;
        if (dict5.get(a[4], -1) == -1):
            dict5[a[4]] = cont5;
            cont5 = cont5 + 1;
        if (dict6.get(a[5], -1) == -1):
            dict6[a[5]] = cont6;
            cont6 = cont6 + 1;
        if (dict7.get(b[0], -1) == -1):
            dict7[b[0]] = cont7;
            cont7 = cont7 + 1;

        valuesTrain[i][0]=  dict1.get(a[0]);
        valuesTrain[i][1] = dict2.get(a[1]);
        valuesTrain[i][2] = dict3.get(a[2]);
        valuesTrain[i][3] = dict4.get(a[3]);
        valuesTrain[i][4] = dict5.get(a[4]);
        valuesTrain[i][5] = dict6.get(a[5]);
        valueslabel[i][(int)(dict7.get(b[0]))]=1;
        #print (valuesTrain[i][0],valuesTrain[i][1],valuesTrain[i][2],valuesTrain[i][3],valuesTrain[i][4],valuesTrain[i][5], dict7.get(b[0]) )
        coord.request_stop()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(file_length_test):
        cx = sess.run([test_dataset])
        #print (cx[0][0],cx[0][1],cx[0][2],cx[0][3],cx[0][4],cx[0][5])
        valuesTest[i][0] = dict1.get(cx[0][0]);
        valuesTest[i][1] = dict2.get(cx[0][1]);
        valuesTest[i][2] = dict3.get(cx[0][2]);
        valuesTest[i][3] = dict4.get(cx[0][3]);
        valuesTest[i][4] = dict5.get(cx[0][4]);
        valuesTest[i][5] = dict6.get(cx[0][5]);
        #print(valuesTest[i][0], valuesTest[i][1], valuesTest[i][2], valuesTest[i][3], valuesTest[i][4],valuesTest[i][5])
        coord.request_stop()

#print (valuesTrain);
graph = tf.Graph()
with graph.as_default():
    # Input dataset
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, inputs))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(valuesTest)

    # Variables
    weights1 = tf.Variable(
        tf.truncated_normal([inputs, hidden1_units]))
    biases1 = tf.Variable(tf.zeros([hidden1_units]))
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    weights2 = tf.Variable(
        tf.truncated_normal([hidden1_units, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))

    def feedforward(dataset):
        h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
        return tf.matmul(h1, weights2) + biases2

    logits = feedforward(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data
    train_prediction = tf.nn.softmax(logits)
    #valid_prediction = tf.nn.softmax(feedforward(tf_valid_dataset))
    test_prediction = tf.nn.softmax(feedforward(tf_test_dataset))

valuesTrain=np.array(valuesTrain)
valueslabel=np.array(valueslabel)


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initiallized')

    for step in range(num_steps):
        offset = (step * batch_size) % (1724 - batch_size)
        # Generate a minibatch
        batch_data = valuesTrain[offset:(offset + batch_size), :]
        batch_labels = valueslabel[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                        feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('\tMinibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    print (test_prediction.eval())
    predictions_values= test_prediction.eval()
    for i in range(file_length_test):
        id=np.argmax(predictions_values[i])
        for key in dict7:
            if(dict7[key]==id):
                print ('solucion ',i,key);
