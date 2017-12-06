#__author__ = 'qiangzha'
import sys, os, os.path as path
import pdb

#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nltk.corpus import stopwords
from utils.load_word_embeddings import LoadEmbeddings
from utils.data_helpers import sent2tokens_wostop
from scipy import spatial
import numpy as np
stoplist = set(stopwords.words('english'))

def avg_feature_vector(sent, model, num_features):
        #function to average all words vectors in a given paragraph
        words = sent2tokens_wostop(sent, stoplist)
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = len(words)

        for word in words:
            if word.isdigit():
                continue
            if model.isKnown(word):
                featureVec = np.add(featureVec, model.word2embedd(word))
            else:
                featureVec = np.add(featureVec, model.word2embedd(u"unknown"))

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec

#add this to pad sequences
def len_feature_vector(sent, max_length,model, embedding_size):
    # function to take max_length words vectors in a given paragraph
    words = sent2tokens_wostop(sent, stoplist)
    featureVec = []
    # Use this zero vector when padding sequences.
    zero_vector = [0] * embedding_size

    if len(words)>=max_length:
        copy_words = words[:max_length]
        for word in copy_words:
            if word.isdigit():
                continue
            if model.isKnown(word):
                featureVec.append(model.word2embedd(word))
            else:
                featureVec.append(model.word2embedd(u"unknown"))
    else:
        for word in words:
            if word.isdigit():
                continue
            if model.isKnown(word):
                featureVec.append(model.word2embedd(word))
            else:
                featureVec.append(model.word2embedd(u"unknown"))
        diff = max_length - len(words)
        for i in range(diff):
            featureVec.append(zero_vector)
    return np.asarray(featureVec)

def fast_cosine(u, v):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio

def avg_embedding_similarity(embeddings, embedding_size, sent1, sent2):
    #print("Calculating similarity for: " + sent1 + "\n and\n" + sent2)
    v1 = avg_feature_vector(sent1, model=embeddings, num_features=embedding_size)
    v2 = avg_feature_vector(sent2, model=embeddings, num_features=embedding_size)
    cosine_distance = fast_cosine(v1,v2)
    #cosine_distance = spatial.distance.cosine(v1, v2)
    #score =  1 - cosine_distance
    #print("Score = " + str(score))
    return v1, v2, cosine_distance

if __name__ == "__main__":
    sent1 = "United States of America"
    sent2 = "USA"

    # configure path and get embedding
    embeddings_path = path.dirname(path.dirname(path.abspath(__file__))) + "/embeddings"
    embeddPath = os.path.normpath("%s/google_news/GoogleNews-vectors-negative300.bin.gz" % (embeddings_path))
    embeddData = os.path.normpath("%s/google_news/embedded_data/" % (embeddings_path))
    vocab_size = 3000000
    embedding_size = 300

    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size, embedding_size=embedding_size)
    v1, v2, cosine_distance = avg_embedding_similarity(embeddings, embedding_size, sent1, sent2)
    print(cosine_distance)