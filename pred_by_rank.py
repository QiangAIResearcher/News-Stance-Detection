from score import *
from util_by_rank import *
import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt

# Prompt for mode
mode = input('mode (load / train)? ')


# Set file names
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "competition_test_stances.csv"
file_test_bodies = "competition_test_bodies.csv"
file_predictions = 'predictions_test.csv'


# Initialise hyperparameters
r = random.Random()
lim_unigram = 3000
hidden_size = 100
train_keep_prob = 0.3
l2_alpha = 0.001
learn_rate = 0.0001
clip_ratio = 5
batch_size_train = 128
epochs = 1000
classes_size = 4

# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)


# Process data sets
train_set,train_mean,train_stances,train_stances_false,bow_vectorizer,tfreq_vectorizer,tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
n_train = len(train_set)
input_size = len(train_set[0])
test_set,test_stances,test_stance_false = \
    pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, train_mean)


# Define model

# Create placeholders
with tf.name_scope('inputs'):
    input_pl = tf.placeholder(tf.float32, [None, input_size], name='unigram_features')
    stances_true_pl = tf.placeholder(tf.int32, [None], 'true_stances')
    stances_false_pl = tf.placeholder(tf.int32,[None,classes_size-1],name='false_stances')#
    keep_prob_pl = tf.placeholder(tf.float32,name='keep_prob')

# Define multi-layer perceptron
with tf.name_scope('layers'):
    with tf.name_scope('hidden_layer'):
        wx_b = tf.contrib.layers.linear(input_pl, hidden_size)
        tf.summary.histogram('wx_b', wx_b)

        # batch normalization
        wx_b_mean,wx_b_var = tf.nn.moments(wx_b,axes=[0])
        scale = tf.Variable(tf.ones([hidden_size]))
        shift = tf.Variable(tf.zeros([hidden_size]))
        epilson = 0.001
        wx_b_normalization = tf.nn.batch_normalization(wx_b,wx_b_mean,wx_b_var,shift,scale,epilson)

        hidden_layer = tf.nn.dropout(tf.nn.relu(wx_b_normalization), keep_prob=keep_prob_pl)#wx_b_normalization

    with tf.name_scope('score_layer'):
        W_class = tf.Variable(tf.random_uniform([hidden_size, classes_size]), name='class_weight')
        tf.summary.histogram('class_weight', W_class)
        s_theta = tf.matmul(hidden_layer, W_class, name='score')
        s_theta_true = tf.gather(tf.reshape(s_theta, [-1], name='reshape_true'), stances_true_pl, name='s_theta_true')

        s_theta_false_temp = tf.reshape(tf.gather(tf.reshape(s_theta, [-1], name='reshape_false_temp_innner'), \
                                                  stances_false_pl, name='gather_false_temp'), \
                                        [-1, classes_size - 1], name='reshape_false_temp_outer')
        s_theta_false = tf.reduce_max(s_theta_false_temp, reduction_indices=[1], name='s_theta_false')# find the maxmimum in each row

        tf.summary.histogram('score_true', s_theta_true)
        tf.summary.histogram('score_false', s_theta_false)

# Define overall loss
with tf.name_scope('loss'):
    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha
    rank_loss = tf.maximum(0., 1. - s_theta_true + s_theta_false)
    loss = tf.reduce_sum(rank_loss+ l2_loss)
tf.summary.scalar('loss', loss)

# Define prediction
with tf.name_scope('prediction'):
    predict = tf.argmax(s_theta, 1)
tf.summary.histogram('prediction', predict)

if mode == 'train':
    # Train model
    early_stop = True
    best_loss = 1000000
    best_epoch = 0
    stopping_step = 0
    early_stopping_step = 11

    val_loss_list = []

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/',sess.graph)
        sess.run(tf.global_variables_initializer())
        # Create a saver object which will save all the variables
        saver = tf.train.Saver()

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        train_loss_list = []
        train_score_list = []
        test_score_list = []
        val_input = test_set
        val_stance = [test_stances[i]+i*4 for i in range(len(test_stances))]
        val_stance_false = [np.array(test_stance_false[i])+i*4 for i in range(len(test_stance_false))]
        val_feed_dict={input_pl:val_input,stances_true_pl:val_stance,stances_false_pl:val_stance_false,keep_prob_pl:1}
        for epoch in range(epochs):
            print(str(epoch)+'/'+str(epochs))
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_input = [train_set[i] for i in batch_indices]

                batch_stances = [train_stances[i] for i in batch_indices]
                batch_stances_adjust = [batch_stances[i]+i*4 for i in range(len(batch_stances))]

                batch_stances_false = [train_stances_false[i] for i in batch_indices]
                batch_stances_false_adjust = [np.array(batch_stances_false[i])+i*4 for i in range(len(batch_stances_false))]

                batch_feed_dict = {input_pl:batch_input, \
                                   stances_true_pl: batch_stances_adjust,\
                                   stances_false_pl:batch_stances_false_adjust, \
                                   keep_prob_pl: train_keep_prob}

                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss

                results = sess.run(merged,feed_dict=batch_feed_dict)
                writer.add_summary(results,i+epoch*(n_train // batch_size_train))

            train_loss_list.append(total_loss)

            # early stopping
            if early_stop == True:
                _, val_loss = sess.run([opt_op, loss], feed_dict=val_feed_dict)
                val_loss_list.append(val_loss)
                if (val_loss < best_loss):# should use the validation loss instead of the train batch loss
                    stopping_step = 0
                    best_loss = val_loss
                    best_epoch = epoch
                    saver.save(sess, './model/best-rank-model-2')
                else:
                    stopping_step += 1
                if stopping_step >= early_stopping_step:
                    print("Early stopping is trigger at epoch: {} loss:{}".format(best_epoch, best_loss))
                    break

        # Predict on training data
        train_feed_dict = {input_pl: train_set, keep_prob_pl: 1.0}
        train_pred = sess.run(predict, feed_dict=train_feed_dict)
        
        # Predict on test data
        test_feed_dict = {input_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
        
        #plt.plot(train_loss_list,label='train_loss')
        plt.plot(val_loss_list, label='val_loss')
        plt.show()

# Load model
if mode == 'load':
    print("\nloading model...")
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './model/best-rank-model')

        # Predict on training data
        train_feed_dict = {input_pl: train_set, keep_prob_pl: 1.0}
        train_pred = sess.run(predict, feed_dict=train_feed_dict)

        #Predict on test data
        test_feed_dict = {input_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

report_score([LABELS[e] for e in test_stances], [LABELS[e] for e in test_pred])
# Save predictions
#save_predictions(test_pred, file_predictions)
