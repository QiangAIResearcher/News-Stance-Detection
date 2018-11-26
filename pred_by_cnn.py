import sys, os, os.path as path
import numpy as np
import collections
from nltk.corpus import stopwords
from utils.util import load_file,save_file,generate_test_training_set,decide_stance,max_length_bodies,max_length_headlines
from utils.articles import Articles
from utils.scorer import report_score

import tensorflow as tf

stoplist = set(stopwords.words('english'))

def load_emergent_datasets(validation_size=500):

    feature_path = path.dirname(path.abspath(__file__)) + "/features"

    test_headlines = load_file(feature_path, "headline_embeddings_test_len.npy")
    train_headlines = load_file(feature_path, "headline_embeddings_train_len.npy")
    test_bodies = load_file(feature_path, "body_embeddings_test_len.npy")

    train_bodies = load_file(feature_path, "body_embeddings_train_len.npy")
    test_stances = load_file(feature_path, "stances_test_len.npy")
    train_stances = load_file(feature_path, "stances_train_len.npy")

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
    #validation_cosine = train_cosine[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_headlines = train_headlines[validation_size:]
    train_bodies = train_bodies[validation_size:]
    #train_cosine = train_cosine[validation_size:]
    train_labels = train_labels[validation_size:]

    train = Articles(train_headlines, train_bodies,train_labels)
    validation = Articles(validation_headlines,validation_bodies,validation_labels)
    test = Articles(test_headlines, test_bodies,test_labels)

    return train, validation,test

def compute_accuracy(v_x1, v_x2, v_y):
    global prediction
    # input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={headline_input: test.headlines, body_input: test.bodies})
    # find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    # calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # get input content
    result = sess.run(accuracy, feed_dict={headline_input: v_x1,body_input:v_x2, onehot_labels: v_y})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf .Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1] , padding='SAME')

def max_pool_2x2(x):
    return tf.nn .max_pool(x, ksize=[1, 2, 2, 1] , strides= [1, 2, 2, 1] ,padding='SAME')

if __name__ == "__main__":
    ######################################
    ##          define parameters       ##
    ######################################
    headline_hidden_size = 200
    body_hidden_size = 200

    batch_size = 32
    embedding_size = 300  # try a larger embedding
    num_classes = 3
    epochs = 10
    learn_rate = 0.001
    l2_alpha = 0.00001
    clip_ratio = 5
    final_state_size = 50
    target_size = 3
    batch_size_train = 50
    train_keep_prob = 0.6
    region_size = 3

    # load data
    train, validation, test = load_emergent_datasets()

    ######################################
    ##           define graph           ##
    ######################################

    # define placeholders
    headline_input = tf.placeholder(tf.float32, [None, max_length_headlines,embedding_size,1])
    body_input = tf.placeholder(tf.float32, [None, max_length_bodies,embedding_size,1])
    onehot_labels = tf.placeholder(tf.float32, [None,target_size])
    keep_prob = tf. placeholder(tf.float32)

    # network structure:
    w_conv1 = weight_variable([region_size, embedding_size, 1, 10])
    b_conv1 = bias_variable([10])
    h_conv1_h = tf.nn.relu(conv2d(headline_input, w_conv1) + b_conv1)
    h_pool1_h = max_pool_2x2(h_conv1_h)
    h_conv1_b = tf.nn.relu(conv2d(body_input, w_conv1) + b_conv1)
    h_pool1_b = max_pool_2x2(h_conv1_h)

    w_conv2 = weight_variable([region_size, embedding_size, 10, 100])
    b_conv2 = bias_variable([100])
    h_conv2_h = tf.nn.relu(conv2d(h_pool1_h, w_conv2) + b_conv2)
    h_pool2_h = max_pool_2x2(h_conv2_h)
    h_conv2_b = tf.nn.relu(conv2d(h_pool1_b, w_conv2) + b_conv2)
    h_pool2_b = max_pool_2x2(h_conv2_h)

    h_concat = tf.concat([h_pool2_h,h_pool2_b],0) # hang pin jie

    W_fc1 = weight_variable([30 * 75 * 100, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_concat, [-1, 30 * 75 * 100])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([128, 3])
    b_fc2 = bias_variable([3])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_mean(-tf.reduce_sum(onehot_labels * tf.log(prediction), reduction_indices=[1]) + l2_loss)

    # Define prediction
    predict = tf.argmax(prediction, 1)

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
    for i in range(10000):
        # get batch to learn easily
        batch_x1, batch_x2, batch_y = train.next_batch_2d(batch_size_train)
        feed_dict = {headline_input: batch_x1, body_input: batch_x2, onehot_labels: batch_y, keep_prob: train_keep_prob}
        _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
        total_loss += current_loss
        if i % 50 == 0:
            print(str(i) + " : " + str(compute_accuracy(validation.headlines, validation.bodies,validation.labels)))

    # sess = tf.Session()
    print("Test accuracy : " + str(compute_accuracy(test.headlines,test.bodies, test.labels)))
    # input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={headline_input: test.headlines,body_input:test.bodies})
    # find how many right
    with tf.Session():
        predicted = tf.argmax(y_pre, 1).eval()  # transoform from tensor to np array
        actual = tf.argmax(test.labels, 1).eval()
        LABELS = ['agree', 'disagree', 'discuss']
        report_score([LABELS[e] for e in actual], [LABELS[e] for e in predicted])


