import numpy as np
import pandas as pd
from csv import DictReader
import os,os.path as path
from utils.load_word_embeddings import LoadEmbeddings
from utils.average_word2vec import avg_feature_vector,len_feature_vector

#character whitelist
chars = set([chr(i) for i in range(32,128)])
max_length_headlines = 20
max_length_bodies = 100  # changed from 700

def configure():

    #set up some values for stances
    stances = {'agree':0, 'disagree':1, 'discuss':2,'unrelated':3}

    _data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data')
    feature_path = path.dirname(os.path.dirname(path.abspath(__file__))) + "/features"

    # configure path and get embedding
    embeddings_path = os.path.dirname(os.path.dirname(path.abspath(__file__))) + "/embeddings"
    embeddPath = os.path.normpath("%s/google_news/GoogleNews-vectors-negative300.bin.gz" % (embeddings_path))
    embeddData = os.path.normpath("%s/google_news/embedded_data/" % (embeddings_path))

    vocab_size = 3000000
    embedding_size = 300

    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size,
                                embedding_size=embedding_size)

    return _data_folder,feature_path, vocab_size, embedding_size, embeddings

def get_dataset(_data_folder, filename='url-versions-2015-06-14.csv'):
    print ("reading dataset")
    """
    folder = os.path.join(_data_folder, filename)
    return pd.DataFrame.from_csv(folder)
    """
    rows = []
    with open(os.path.join(_data_folder,filename),'r') as table:#,encoding='utf-8'
        dict= DictReader(table)

        for line in dict:
            rows.append(line)

    print("Total samples: " + str(len(rows)))
    return rows

def decide_stance(headline_stance,body_stance):

    if (body_stance == 'for' and headline_stance == 'for')\
            or (body_stance == 'against' and headline_stance == 'against'):
        valid_stance = 0 #'agree'
    elif (body_stance== 'for' and headline_stance == 'against')\
            or (body_stance == 'against' and headline_stance == 'for'):
        valid_stance = 1 #'disagree'
    elif body_stance == 'observing' or headline_stance == 'observing':
        valid_stance = 2 #'discuss'
    else:
        print ("weird invalid stances")

    return valid_stance

def gen_stance_list(stances):
    agree_list = []
    disagree_list = []
    discuss_list = []

    for i_th in range(0,len(stances)):
        if stances[i_th] == 0:
            agree_list.append(stances[i_th])
        elif stances[i_th] == 1:
            disagree_list.append(stances[i_th])
        elif stances[i_th] == 2:
            discuss_list.append(stances[i_th])
        else:
            print ("weird invalid stances")

    print("Total agree: " + str(len(agree_list)))  # 2138
    print("Total disagree: " + str(len(disagree_list)))  # 35
    print("Total discuss: " + str(len(discuss_list)))  # 2818

    return agree_list,disagree_list,discuss_list

def generate_data_embeddings(rows,embeddings,embedding_size):

    headlines = []
    bodies = []
    stances = []

    # check stances and construct data list
    for i,row in enumerate(rows):
        v1 = avg_feature_vector(row['articleHeadline'], model=embeddings, num_features=embedding_size)
        v2 = avg_feature_vector(row['articleBody'], model=embeddings, num_features=embedding_size)
        headlines.append(v1)
        bodies.append(v2)
        stances.append(decide_stance(row['articleHeadlineStance'],row['articleStance']))

    agree_idx_list, disagree_idx_list, discuss_idx_list = gen_stance_list(stances)

    return headlines,bodies, stances

def generate_length_embeddings(rows,embeddings, embedding_size, max_length_headlines, max_length_bodies):
    headlines = []
    bodies = []
    stances = []

    # check stances and construct data list
    for i, row in enumerate(rows):
        v1 = len_feature_vector(row['articleHeadline'], max_length_headlines,model=embeddings, embedding_size=embedding_size)
        v2 = len_feature_vector(row['articleBody'], max_length_bodies, model=embeddings, embedding_size=embedding_size)
        headlines.append(v1)
        bodies.append(v2)
        stances.append(decide_stance(row['articleHeadlineStance'], row['articleStance']))

    return headlines, bodies, stances

def generate_test_training_set(rows, test_set_fraction=0.1):
    articleId_list = []
    for i, row in enumerate(rows):
        articleId_list.append(row['articleId'])

    article_ids = np.array(list(set(articleId_list)))
    article_ids_rand = np.random.permutation(article_ids)
    article_ids_test = article_ids_rand[:int(len(article_ids_rand) * test_set_fraction)]
    article_ids_train = set(article_ids_rand).difference(article_ids_test)

    test_rows =  [row for row in rows if row['articleId'] in article_ids_test]
    train_rows = [row for row in rows if row['articleId'] in article_ids_train]

    return test_rows, train_rows

def save_file(filepath,filename,variables):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    file_short = os.path.normpath("%s/%s" % (filepath, filename))
    np.save(file_short,variables)

def load_file(filepath,filename):
    if not os.path.exists(filepath):
        print("non-existent files")

    file_short = os.path.normpath("%s/%s" % (filepath, filename))
    variables = np.load(file_short)
    return variables

if __name__ == "__main__":
    print("start...")
    _data_folder, feature_path, vocab_size, embedding_size, embeddings = configure()
    # read data
    clean_rows = load_file(_data_folder, "url-versions-2015-06-14-CleanQZ.npy")

    # generate training and test data
    test_rows, train_rows = generate_test_training_set(clean_rows)

    debug = True
    if debug == True:
        # generate embeddings of docs by summing up all word embedding
        test_headlines, test_bodies, test_stances = generate_data_embeddings(test_rows, embeddings, embedding_size)
        train_headlines, train_bodies, train_stances = generate_data_embeddings(train_rows, embeddings, embedding_size)

        # generate embeddings of docs with fixed numbers of words
        #test_headlines, test_bodies, test_stances = \
        #    generate_length_embeddings(test_rows, embeddings, embedding_size,max_length_headlines, max_length_bodies)
        #train_headlines, train_bodies, train_stances = \
        #    generate_length_embeddings(train_rows, embeddings,embedding_size, max_length_headlines,max_length_bodies)

        save_file(feature_path, "headline_embeddings_test", test_headlines)
        save_file(feature_path, "headline_embeddings_train", train_headlines)
        save_file(feature_path, "body_embeddings_test", test_bodies)
        save_file(feature_path, "body_embeddings_train", train_bodies)
        save_file(feature_path, "stances_test", test_stances)
        save_file(feature_path, "stances_train", train_stances)
    else:
        test_headlines = load_file(feature_path, "headline_embeddings_test.npy")
        train_headlines = load_file(feature_path, "headline_embeddings_train.npy")
        test_bodies = load_file(feature_path, "body_embeddings_test.npy")
        train_bodies = load_file(feature_path, "body_embeddings_train.npy")
        test_stances = load_file(feature_path, "stances_test.npy")
        train_stances = load_file(feature_path, "stances_train.npy")

    print ("hello")#7112