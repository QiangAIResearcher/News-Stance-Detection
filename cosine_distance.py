import os, os.path as path
import numpy as np

from utils.doc2vec import fast_cosine
from utils.util import configure,get_dataset,save_file,load_file

if __name__ == "__main__":
    _data_folder, feature_path, vocab_size, embedding_size, embeddings = configure()

    train_headlines = load_file(feature_path, "headline_embeddings_train.npy")
    train_bodies = load_file(feature_path, "body_embeddings_train.npy")
    #train_stances = load_file(feature_path, "stances_train.npy")
    test_headlines = load_file(feature_path, "headline_embeddings_test.npy")
    test_bodies = load_file(feature_path, "body_embeddings_test.npy")
    #test_stances = load_file(feature_path, "stances_test.npy")

    train_cos_list=[]
    test_cos_list =[]
    # check if where is some invalid cosine
    for idx in range(len(train_headlines)):
        cosine_distance = fast_cosine(train_headlines[idx],train_bodies[idx])
        if cosine_distance <= 1.0+1e-4 and cosine_distance >= 1e-4:
            train_cos_list.append(cosine_distance)
        #else:
        #    invalid_cosine_idx_list.append(valid_idx_list[idx])
        #    print ('unvalid cosine ' + str(valid_idx_list[idx]) + ': ' + str(cosine_distance))
    for idx in range(len(test_headlines)):
        cosine_distance = fast_cosine(test_headlines[idx],test_bodies[idx])
        if cosine_distance <= 1.0+1e-4 and cosine_distance >= 1e-4:
            test_cos_list.append(cosine_distance)
    # all have been checkd to produce valid cosine distance
    print ('number of valid cosine distance: ' + str(len(train_cos_list)+len(test_cos_list)))

    # save features
    save_file(feature_path, filename='cosine_train',variables=train_cos_list)
    save_file(feature_path, filename='cosine_test', variables=test_cos_list)
    #ddd = load_file(feature_path, "valid_cosine.npy")

    #save_file(filepath=feature_path, filename='valid_cosine',variables=np.array([valid_idx_list, valid_cos_list], dtype=object))
    #myarray = np.fromfile('%s/%s.dat' % (feature_path, 'valid_cosine'), dtype=np.double)
    #idx = myarray[:len(valid_idx_list)].astype(int)
    #cos = myarray[len(valid_idx_list):]

    print ("hello")