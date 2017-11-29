#__author__ = 'qiangzha'
'''
This script get data from the .CSV file and parse components
'''
import os
import nltk
#import re
import numpy as np
#from sklearn import feature_extraction
#from nltk.stem.porter import PorterStemmer
from csv import DictReader
import sys, os, os.path as path
from load_word_embeddings import LoadEmbeddings
from doc2vec import avg_embedding_similarity
from itertools import chain

#character whitelist
chars = set([chr(i) for i in range(32,128)])
#set up some values for stances
#stances = {'agree':0, 'disagree':1, 'discuss':2,'unrelated':3}
VALID_STANCE_LABELS = ['for', 'against','observing']

_data_folder = os.path.join(os.path.dirname(__file__),'data')

def get_dataset(filename='url-versions-2015-06-14.csv'):
    print ("reading dataset")
    rows = []
    with open(os.path.join(_data_folder,filename),'r') as table:#,encoding='utf-8'
        r= DictReader(table)

        for line in r:
            rows.append(line)

    print("Total samples: " + str(len(rows)))
    return rows


def split_data(data):
    y = data.articleHeadlineStance
    x = data[['claimHeadline','articleHeadline','claimId','articleId']]
    return x,y
'''
def clean_sentence(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+',s,flags=re.UNICODE)).lower()

_wnl = nltk.WordNetLemmatizer()
def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def remove_stopwords(l):
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

_stemmer = PorterStemmer()

def get_stem(w):
    return _stemmer.stem(w)
    
def gen_load_file(filename,idx_list,article_bodies):
    if not os.path.isfile(filename):
        text = ''
        for i in idx_list:
            text = text + ' ' + article_bodies[i]
        with open (filename,"w") as txt_file:
            txt_file.write(text)

    else:
        with open (filename,"r") as txt_file:
            text = txt_file.read()
    return text
'''
def generate_test_training_set(data, test_set_fraction=0.2):
    """
    Splits the given data into mutually exclusive test
    and training sets using the claim ids.
    :param data: DataFrame containing the data
    :param test_set_fraction: percentage of data to reserve for test
    :return: a tuple of DataFrames containing the test and training data
    """
    claim_ids = np.array(list(set(data.claimId.values)))
    claim_ids_rand = np.random.permutation(claim_ids)
    claim_ids_test = claim_ids_rand[:len(claim_ids_rand) * test_set_fraction]
    claim_ids_train = set(claim_ids_rand).difference(claim_ids_test)
    test_data = data[data.claimId.isin(claim_ids_test)]
    train_data = data[data.claimId.isin(claim_ids_train)]
    return test_data, train_data


def gen_idx_list(headline_stances,article_stances,idx_list):
    agree_list = []
    disagree_list = []
    discuss_list = []

    for i_th in range(0,len(headline_stances)):
        if (article_stances[i_th] == 'for' and headline_stances[i_th] == 'for')\
                or (article_stances[i_th] == 'against' and headline_stances[i_th] == 'against'):
            agree_list.append(idx_list[i_th])
        elif (article_stances[i_th] == 'for' and headline_stances[i_th] == 'against')\
                or (article_stances[i_th] == 'against' and headline_stances[i_th] == 'for'):
            disagree_list.append(idx_list[i_th])
        elif article_stances[i_th] == 'observing' or headline_stances[i_th] == 'observing':
            discuss_list.append(idx_list[i_th])
        else:
            print ("weird invalid stances")

    print("Total agree: " + str(len(agree_list)))  # 2138
    print("Total disagree: " + str(len(disagree_list)))  # 35
    print("Total discuss: " + str(len(discuss_list)))  # 2818

    return agree_list,disagree_list,discuss_list

def save_file(filepath,filename,variables):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file_short = os.path.normpath("%s/%s.dat" %(filepath, filename))

    fp = np.memmap(file_short, dtype=np.double, mode='w+', shape=variables.shape)
    fp[:,:] = variables[:,:]
    del fp


if __name__ == "__main__":

    df = get_dataset()

    data_path = path.dirname(path.abspath(__file__)) + "/embeddings"
    embeddPath = os.path.normpath("%s/google_news/GoogleNews-vectors-negative300.bin.gz" % (data_path))
    embeddData = os.path.normpath("%s/google_news/embedded_data/" % (data_path))
    vocab_size = 3000000
    embedding_size = 300

    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size,
                                embedding_size=embedding_size)
    headlines = []
    headline_stances = []
    articles = []
    article_stances = []
    valid_idx_list = []
    valid_cos_list = []

    invalid_stance = []
    invalid_cosine = []

    # check stances and construct data list
    for i,row in enumerate(df):
        if row['articleHeadlineStance'] in VALID_STANCE_LABELS and row['articleStance'] in VALID_STANCE_LABELS:
            v1, v2, cosine_distance = avg_embedding_similarity(embeddings, embedding_size, row['articleHeadline'],
                                                               row['articleBody'])
            if cosine_distance < 1.0:
                headlines.append(row['articleHeadline'])
                headline_stances.append(row['articleHeadlineStance'])
                articles.append(row['articleBody'])
                article_stances.append(row['articleStance'])
                valid_idx_list.append(i)
                valid_cos_list.append(cosine_distance)
                print('valid ' + str(i) + ' cosine distance: ' + str(cosine_distance))
            else:
                invalid_cosine.append(i)
                print ('unvalid cosine ' + str(i))
        else:
            invalid_stance.append(i)
            print ('unvalid stance ' + str(i))
    print ('number of unvalid stance : ' + str(len(invalid_stance)))
    print ('number of unvalid cosine : ' + str(len(invalid_cosine)))
    print ('number of valid samples : '+ str(len(valid_idx_list)))

    agree_idx_list, disagree_idx_list, discuss_idx_list = gen_idx_list(headline_stances, article_stances, valid_idx_list)

    feature_path = path.dirname(path.abspath(__file__)) + "/features"
    save_file(filepath=feature_path, filename='valid_cosine', variables=np.array([valid_idx_list, valid_cos_list],dtype=object))
    myarray = np.fromfile('%s/%s.dat'%(feature_path,'valid_cosine'), dtype=np.double)
    idx = myarray[:len(valid_idx_list)].astype(int)
    cos = myarray[len(valid_idx_list):]

    print ("hello")#7112
