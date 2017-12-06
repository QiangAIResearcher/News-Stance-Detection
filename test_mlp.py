import sys, os, os.path as path
import numpy as np
from utils.util import load_file
from utils.articles import Articles
from utils.scorer import report_score

import tensorflow as tf

def compute_accuracy(v_x, v_y):
    global prediction
    # input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x: v_x})
    # find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    # calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # get input content
    result = sess.run(accuracy, feed_dict={x: v_x, onehot_labels: v_y})
    return result

def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    # calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # add the active hanshu
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs

def load_emergent_datasets(validation_size=500):

    feature_path = path.dirname(path.abspath(__file__)) + "/features"

    train_headlines = load_file(feature_path, "headline_embeddings_train.npy")
    train_bodies = load_file(feature_path, "body_embeddings_train.npy")
    train_cosine = load_file(feature_path, "cosine_train.npy")
    train_stances = load_file(feature_path, "stances_train.npy")

    test_headlines = load_file(feature_path, "headline_embeddings_test.npy")
    test_bodies = load_file(feature_path, "body_embeddings_test.npy")
    test_cosine = load_file(feature_path, "cosine_test.npy")
    test_stances = load_file(feature_path, "stances_test.npy")

    train_labels = []
    test_labels = []
    for stance in train_stances:
        if stance == 0:
            train_labels.append([1, 0, 0])
        elif stance == 1:
            train_labels.append([0, 1, 0])
        else:
            train_labels.append([0, 0, 1])

    for stance in test_stances:
        if stance == 0:
            test_labels.append([1, 0, 0])
        elif stance == 1:
            test_labels.append([0, 1, 0])
        else:
            test_labels.append([0, 0, 1])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    if not 0 <= validation_size <= len(train_labels):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_labels), validation_size))

    validation_headlines = train_headlines[:validation_size]
    validation_bodies = train_bodies[:validation_size]
    validation_cosine = train_cosine[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_headlines = train_headlines[validation_size:]
    train_bodies = train_bodies[validation_size:]
    train_cosine = train_cosine[validation_size:]
    train_labels = train_labels[validation_size:]

    train = Articles(train_headlines, train_bodies,train_labels)
    validation = Articles(validation_headlines,validation_bodies,validation_labels)
    test = Articles(test_headlines, test_bodies,test_labels)

    return train, validation, test

if __name__ == '__main__':

    # Initialise hyperparameters
    input_size = 600
    target_size = 3
    hidden_size = 100
    train_keep_prob = 0.6
    l2_alpha = 0.00001
    learn_rate = 0.01
    clip_ratio = 5
    batch_size_train = 50
    epochs = 90

    # load data
    train,validation,test = load_emergent_datasets()


    ######################################
    ##           define graph           ##
    ######################################
    # define placeholders

    x = tf.placeholder(tf.float32, [None, input_size]) # embedding + embedding #+ distance (cosine)
    onehot_labels = tf.placeholder(tf.float32,[None,target_size])
    keep_prob = tf.placeholder(tf.float32)

    # Define multi-layer perceptron
    hidden_layer = add_layer(x, input_size, hidden_size, activation_function=tf.nn.softmax)
    prediction = add_layer(hidden_layer, hidden_size, target_size, activation_function=tf.nn.softmax)

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    '''
    # your class weights
    class_weights = tf.constant([[1.0, 8.0, 1.0]])
    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=prediction)
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses+ l2_loss)
    '''
    #loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=y,logits=prediction,pos_weight=weights) + l2_loss)
    loss = tf.reduce_mean(-tf.reduce_sum(onehot_labels * tf.log(prediction), reduction_indices=[1])+ l2_loss)

    # Define prediction
    softmaxed_prediction = tf.nn.softmax(prediction)
    predict = tf.argmax(softmaxed_prediction, 1)

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    sess = tf.Session()
    # init all variables
    sess.run(tf.global_variables_initializer())
    total_loss = 0
    # start training
    for i in range(30000):
        # get batch to learn easily
        batch_x, batch_y = train.next_batch(batch_size_train)
        feed_dict = {x: batch_x, onehot_labels: batch_y, keep_prob: train_keep_prob}
        _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
        total_loss += current_loss
        if i % 50 == 0:
            print(str(i) + " : " + str(compute_accuracy(validation.input, validation.labels)))

    #sess = tf.Session()
    print( "Test accuracy : " + str(compute_accuracy(test.input, test.labels)))
    # input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x: test.input})
    # find how many right
    with tf.Session():
        predicted = tf.argmax(y_pre, 1).eval()# transoform from tensor to np array
        actual = tf.argmax(test.labels, 1).eval()
        LABELS = ['agree', 'disagree', 'discuss']
        report_score([LABELS[e] for e in actual], [LABELS[e] for e in predicted])
