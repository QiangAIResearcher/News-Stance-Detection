from __future__ import division
# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
from csv import DictReader
from csv import DictWriter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from feature_engineering import *
import tensorflow as tf


# Initialise global variables
label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
label_num = [0,1,2,3]
stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

# Define data class
class FNCData:

    """

    Define class for Fake News Challenge data

    """

    def __init__(self, file_instances, file_bodies):

        # Load data
        self.instances = self.read(file_instances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Headline'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Headline']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    def read(self, filename):

        """
        Read Fake News Challenge data from CSV file

        Args:
            filename: str, filename + extension

        Returns:
            rows: list, of dict per instance

        """

        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows


# Define relevant functions
def pipeline_train(train, test, lim_unigram):

    """

    Process train set, create relevant vectorizers

    Args:
        train: FNCData object, train set
        test: FNCData object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        train_set: list, of numpy arrays
        train_stances: list, of ints
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    """

    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    body_ids = []
    id_ref = {}
    train_set = []
    train_stances = []
    train_stance_false = []
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    test_body_ids = []
    head_tfidf_track = {}
    body_tfidf_track = {}
    head_all = []
    body_all = []
    # Identify unique heads and bodies
    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            heads.append(head)
            heads_track[head] = 1
        if body_id not in bodies_track:
            bodies.append(train.bodies[body_id])
            bodies_track[body_id] = 1
            body_ids.append(body_id)

    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in test_heads_track:
            test_heads.append(head)
            test_heads_track[head] = 1
        if body_id not in test_bodies_track:
            test_bodies.append(test.bodies[body_id])
            test_bodies_track[body_id] = 1
            test_body_ids.append(body_id)

    # Create reference dictionary
    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
        fit(heads + bodies + test_heads + test_bodies)  # Train and test sets

    skip_unrelated = 0
    skip_discuss = 0
    # Process train set
    for instance in train.instances:
        true_stance = label_ref[instance['Stance']]  # number, not string
        if true_stance != 4: # use all samples
            head = instance['Headline']
            body_id = instance['Body ID']
            body = train.bodies[body_id]
            head_all.append(head)
            body_all.append(body)
            head_tf = tfreq[id_ref[head]].reshape(1, -1)
            body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
            if head not in head_tfidf_track:
                head_tfidf = tfidf_vectorizer.transform([head]).toarray()
                head_tfidf_track[head] = head_tfidf
            else:
                head_tfidf = head_tfidf_track[head]
            if body_id not in body_tfidf_track:
                body_tfidf = tfidf_vectorizer.transform([train.bodies[body_id]]).toarray()
                body_tfidf_track[body_id] = body_tfidf
            else:
                body_tfidf = body_tfidf_track[body_id]
            if (head, body_id) not in cos_track:
                tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
                cos_track[(head, body_id)] = tfidf_cos
            else:
                tfidf_cos = cos_track[(head, body_id)]

            feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])

            add_data = 1
            '''
            if true_stance == 3:
                skip_unrelated += 1
                if skip_unrelated % 10 != 0:
                    add_data = 0
            elif true_stance == 2:
                skip_discuss += 1
                if skip_discuss % 3 != 0:
                    add_data = 0
            elif true_stance == 1:
                for i in range(3): # add 3 times
                    train_set.append(feat_vec)
                    train_stances.append(true_stance)
                    train_stance_false.append([i for i in label_num if i != true_stance])
            '''
            if add_data == 1:
                train_set.append(feat_vec)
                train_stances.append(true_stance)
                train_stance_false.append([i for i in label_num if i != true_stance])

    X_overlap = gen_or_load_feats(word_overlap_features, head_all, body_all, 'features/train_overlap.npy')
    X_refuting = gen_or_load_feats(refuting_features, head_all, body_all, 'features/train_refuting.npy')
    X_polarity = gen_or_load_feats(polarity_features, head_all, body_all, 'features/train_polarity.npy')
    X_hand = gen_or_load_feats(hand_features, head_all, body_all, 'features/train_hand.npy')

    train_features = np.squeeze(np.c_[ X_refuting, X_polarity,X_overlap,X_hand])#
    train_set = np.squeeze(np.c_[train_set,train_features])
    ####### preprocessing
    train_set = np.asarray(train_set)
    train_mean = np.mean(train_set,axis = 0)
    '''
    # it becomes much worse to subtract std
    train_std = np.std(train_set, axis = 0)
    #zero_std_idx = []
    xxx = np.median(train_std)
    num = 0
    for i in range(len(train_std)):
        if train_std[i] <=0.000001: # it's believed to be 0
            print(train_std[i])
            train_std[i] = xxx
            num += 1
            #zero_std_idx.append(i)
    print("number of zero std is: %d" % num)  # 1917
    #np.delete(train_std,zero_std_idx,axis=1) # axis=1 means delete certain columns
    #np.delete(train_set,zero_std_idx,axis=1)
'''
    train_set = (train_set-train_mean)
    '''
    # PCA and whitening
    cov = np.dot(train_set.T, train_set) / train_set.shape[0]  # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)
    train_set_rot_reduced = np.dot(train_set, U[:,:5000])  # decorrelate the data
    # whiten the data: divide by the eigenvalues (which are square roots of the singular values)
    train_set = train_set_rot_reduced / np.sqrt(S[:5000] + 1e-5)
    '''
    return train_set,train_mean, train_stances, train_stance_false, \
           bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, m):

    """
    Process test set
    Args:
        test: FNCData object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()
    Returns:
        test_set: list, of numpy arrays
    """

    # Initialise
    test_set = []
    test_stances = []
    test_stance_false = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}
    head_all = []
    body_all = []

    # Process test set
    for instance in test.instances:
        true_stance = label_ref[instance['Stance']]  # number, not string
        if true_stance != 4:  # use all samples
            head = instance['Headline']
            body_id = instance['Body ID']
            body = test.bodies[body_id]
            head_all.append(head)
            body_all.append(body)
            if head not in heads_track:
                head_bow = bow_vectorizer.transform([head]).toarray()
                head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
                head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
                heads_track[head] = (head_tf, head_tfidf)
            else:
                head_tf = heads_track[head][0]
                head_tfidf = heads_track[head][1]
            if body_id not in bodies_track:
                body_bow = bow_vectorizer.transform([test.bodies[body_id]]).toarray()
                body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
                body_tfidf = tfidf_vectorizer.transform([test.bodies[body_id]]).toarray().reshape(1, -1)
                bodies_track[body_id] = (body_tf, body_tfidf)
            else:
                body_tf = bodies_track[body_id][0]
                body_tfidf = bodies_track[body_id][1]
            if (head, body_id) not in cos_track:
                tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
                cos_track[(head, body_id)] = tfidf_cos
            else:
                tfidf_cos = cos_track[(head, body_id)]

            feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])

            test_set.append(feat_vec)
            test_stances.append(label_ref[instance['Stance']])
            test_stance_false.append([i for i in label_num if i != true_stance])

    X_overlap = gen_or_load_feats(word_overlap_features, head_all, body_all, 'features/test_overlap.npy')
    X_refuting = gen_or_load_feats(refuting_features, head_all, body_all, 'features/test_refuting.npy')
    X_polarity = gen_or_load_feats(polarity_features, head_all, body_all, 'features/test_polarity.npy')
    X_hand = gen_or_load_feats(hand_features, head_all, body_all, 'features/test_hand.npy')

    test_features = np.squeeze(np.c_[X_refuting,X_polarity,X_overlap,X_hand])#
    test_set = np.squeeze(np.c_[test_set, test_features])
    ####### preprocessing
    test_set = np.asarray(test_set)
    test_set = (test_set-m)#/(std+0.0001)
    '''
    PCA and whitening
    cov = np.dot(test_set.T, test_set) / test_set.shape[0]  # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)
    test_set_rot_reduced = np.dot(test_set, U[:, :5000])  # decorrelate the data
    # whiten the data:
    test_set = test_set_rot_reduced / np.sqrt(S[:5000] + 1e-5)
    '''
    return test_set,test_stances,test_stance_false


def load_model(sess):

    """

    Load TensorFlow model

    Args:
        sess: TensorFlow session

    """

    saver = tf.train.Saver()
    saver.restore(sess, './model/model.checkpoint')


def save_predictions(pred, file):

    """

    Save predictions to CSV file

    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension

    """

    with open(file, 'w') as csvfile:
        fieldnames = ['Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for instance in pred:
            writer.writerow({'Stance': label_ref_rev[instance]})
