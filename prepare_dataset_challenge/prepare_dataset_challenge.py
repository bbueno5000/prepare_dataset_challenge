"""
DOCSTRING
"""
import numpy
import tensorflow

NUM_LABELS = 2
BATCH_SIZE = 100

tensorflow.app.flags.DEFINE_string(
    'train', None, 'File containing the training data (labels & features).')
tensorflow.app.flags.DEFINE_string(
    'test', None, 'File containing the test data (labels & features).')
tensorflow.app.flags.DEFINE_integer(
    'num_epochs', 1,
    'Number of examples to separate from the training data for the validation set.')
tensorflow.app.flags.DEFINE_boolean(
    'verbose', False, 'Produce verbose output.')

FLAGS = tensorflow.app.flags.FLAGS

def extract_data(filename):
    """
    Extract numpy representations of the labels and features given rows consisting of:
    label, feat_0, feat_1, ..., feat_n
    """
    labels, fvecs = list(), list()
    for line in file(filename):
        row = line.split(",")
        labels.append(int(row[0]))
        fvecs.append([float(x) for x in row[1:]])
    fvecs_numpy = numpy.matrix(fvecs).astype(numpy.float32)
    labels_numpy = numpy.array(labels).astype(dtype=numpy.uint8)
    labels_onehot = (numpy.arange(NUM_LABELS) == labels_np[:, None]).astype(numpy.float32)
    return fvecs_np,labels_onehot

def main(argv=None):
    verbose = FLAGS.verbose
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test
    train_data,train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)
    train_size,num_features = train_data.shape
    num_epochs = FLAGS.num_epochs
    x = tensorflow.placeholder("float", shape=[None, num_features])
    y_ = tensorflow.placeholder("float", shape=[None, NUM_LABELS])
    test_data_node = tensorflow.constant(test_data)
    W = tensorflow.Variable(tensorflow.zeros([num_features,NUM_LABELS]))
    b = tensorflow.Variable(tensorflow.zeros([NUM_LABELS]))
    y = tensorflow.nn.softmax(tensorflow.matmul(x,W) + b)
    cross_entropy = -tensorflow.reduce_sum(y_*tensorflow.log(y))
    train_step = tensorflow.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tensorflow.equal(tensorflow.argmax(y,1), tensorflow.argmax(y_,1))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, "float"))
    with tensorflow.Session() as s:
        tensorflow.initialize_all_variables().run()
        if verbose:
            print('Initialized.')
            print('Training.')
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print(step,)
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})
            if verbose and offset >= train_size-BATCH_SIZE:
                print()
        if verbose:
            print()
            print('Weight matrix.')
            print(s.run(W))
            print()
            print('Bias vector.')
            print(s.run(b))
            print()
            print('Applying model to first test instance.')
            first = test_data[:1]
            print('Point =', first)
            print('Wx + b = ', s.run(tensorflow.matmul(first, W) + b))
            print('softmax(Wx + b) = ', s.run(tensorflow.nn.softmax(tensorflow.matmul(first, W) + b)))
            print()
        print('Accuracy:', accuracy.eval(feed_dict={x: test_data, y_: test_labels}))

if __name__ == '__main__':
    tensorflow.app.run()
