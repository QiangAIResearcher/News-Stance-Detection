import os
import csv
import numpy as np
from util import get_dataset,save_file

VALID_STANCE_LABELS = ['for', 'against','observing']
_data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data')

def clean_dataset(rows,VALID_STANCE_LABELS):#,_data_folder, filename='url-versions-2015-06-14-cleanQZ.csv'
    clean_rows = []
    for idx, row in enumerate(rows):
        if row['articleHeadlineStance'] in VALID_STANCE_LABELS and row['articleStance'] in VALID_STANCE_LABELS:
            if row['articleHeadline'] != '' and row['articleBody'] != '':
                clean_rows.append(row)
    print ('number of valid data samples : ' + str(len(clean_rows)))
    return clean_rows

if __name__ == "__main__":
    rows = get_dataset(_data_folder)

    # clean data and save it as npy file
    clean_rows = clean_dataset(rows, VALID_STANCE_LABELS)
    save_file(_data_folder, filename='url-versions-2015-06-14-CleanQZ', variables=clean_rows)
