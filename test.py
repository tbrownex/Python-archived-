import tensorflow as tf
import numpy as np
import pandas as pd
import sys

DATA_DIR      = "/home/tom/Dropbox/data/ML/LV/SimpliSafe/"
FILENM        = 'test.csv'
SAVE_DIR      = "/home/tom/ML/tf_checkpoints/"

SELECTED_COLS = ['age_alt', 'income_bins', 'marital_status', 'owner_type',
                 'hv_bins', 'num_hh', 'num_adults', 'num_kids',
                 'net_worth_alt', 'density_bins', 'white', 'black', 'hispanic',
                 'asian', 'jewish', 'indian', 'other', 'Label']

# Load the Test file: split data and labels. Save off the HH_num to be used later for generating a mailing list
def get_data():
    df = pd.read_csv(DATA_DIR+FILENM, sep="|")
    df = df[SELECTED_COLS]
    
    labels = df['Label']
    
    del df['Label']
    data = np.array(df)
    
    print('{}Test file has {:,.0f} rows'.format('\n',df.shape[0]))
    return(data, labels)

if __name__ == "__main__":
    data, labels = get_data()
    
    # get the true response rate to compare to my predictions
    print("labels sum: ", labels.sum())
    print("labels length: ", labels.shape[0])
    true_rr = labels.sum() / labels.shape[0]
    print(true_rr)
    
    # Get the name of the saved model to use
    model = input("Enter the name of the saved model:")
    
    tf.reset_default_graph()
    ckpoint = SAVE_DIR+model
    
    saver = tf.train.import_meta_graph(ckpoint+".meta")
    sess = tf.Session()
    saver.restore(sess, ckpoint)
    
    predictions = sess.run("L2:0", feed_dict={"input:0": data})
    
    K = 100000
    ix = np.argpartition(predictions[:,1], -K)[-K:]
    
    topK  = labels[ix].sum()
    my_rr = topK / K
    print("True Positives: {:.0f} with lift: {:.2%}".format(topK,(my_rr/true_rr-1)))
    sess.close()