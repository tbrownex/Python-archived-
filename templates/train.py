import pandas as pd
import numpy  as np
from scale_split_data import split
import os
import sys
import time
from nn               import run
import jobNumber as job

CLIENT   = 'SimpliSafe/'
TEST_PCT = 0.0
VAL_PCT  = 0.25
LABELS   = np.array([0,1])
DATA_DIR = "/home/tom/Dropbox/data/ML/LV/"
FILENM   = 'train.csv'

# This is for the Summary file
keys = ['l1_size','learning_rate','lambda','weight','batch_size','epochs','activation']

SELECTED_COLS = ['age_alt', 'income_bins', 'marital_status', 'owner_type',
                 'hv_bins', 'num_hh', 'num_adults', 'num_kids',
                 'net_worth_alt', 'density_bins', 'white', 'black', 'hispanic',
                 'asian', 'jewish', 'indian', 'other', 'Label']

# NN hyper-parameters
l1_size       = [10]               # Count of nodes in layer 1
learning_rate = [.0003]
Lambda        = [0.1]              # Regularization parameter
weight        = [300]              # Degree to which Positives are weighted in the loss function
batch_size    = [128]
epochs        = [10]
activation    = ['ReLU']           # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'

def get_data():
    df = pd.read_csv(DATA_DIR+CLIENT+FILENM, sep="|")
    df = df[SELECTED_COLS]
    df = df.sample(frac=0.15)
    
    print('{}Input file has {:,.0f} rows'.format('\n',df.shape[0]))
    print('{:<15}{:,.0f}'.format('R',df.loc[df['Label']==1.0].shape[0]))
    print('{:<15}{:,.0f}'.format('NR',df.loc[df['Label']==0.0].shape[0]))
    return(df)

# Split into train and val
def split_data(df):
    train, val, test = split(df, VAL_PCT, TEST_PCT)
    
    train = train.as_matrix()
    val   = val.as_matrix()
    
    data_dict = {}
    data_dict['train_x'] = train[:,:-1]
    data_dict['val_x']   = val[:,:-1]
    
# convert labels to "one-hot" vectors
    train_y = train[:,-1]
    val_y   = val[:,-1]
        
    data_dict['train_labels'] = (LABELS == train_y[:, None]).astype(np.float32)
    data_dict['val_labels']   = (LABELS == val_y[:,   None]).astype(np.float32)
    
    print("Validation set has {:,.0f} positives out of {:,.0f}".format(np.sum(val[:,-1]), val.shape[0]))
    _=input("Enter to continue")
    return data_dict

def save_results(count, parms, results):
    rec = ''
    for k in keys:
        rec += str(parms[k]) +"|"
    rec += str(results[0]) +"|"
    rec += str(results[1]) +"\n"
    summary.write(rec)

if __name__ == "__main__":
    df = get_data()
    data_dict = split_data(df)
    
    # Prepare the file which holds results of a run
    job_id = int(job.getJob())
    summary = open(DATA_DIR+CLIENT+str(job_id)+'summary_'+str(job_id)+".txt", 'w')
    rec = "|".join(keys)
    rec += "|"+"TP count"+ "|"+"Lift"
    rec = rec+"\n"
    summary.write(rec)
                   
    for x in activation:
        assert x in ['tanh', 'leakyReLU', 'ReLU', 'relu6'], "Invalid Activation: %s" % x
        
    parms = [[a,b,c,d,e,f,g] for a in l1_size
             for b in learning_rate
             for c in Lambda
             for d in weight
             for e in batch_size
             for f in epochs
             for g in activation]
                 
    parm_dict = {}
    count = 1
    start_time = time.time()
    
    loop = 1
    for i in range(loop):
        for x in parms:
            loop_time = time.time()
            parm_dict['l1_size']       = x[0]
            parm_dict['learning_rate'] = x[1]
            parm_dict['lambda']        = x[2]
            parm_dict['weight']        = x[3]
            parm_dict['batch_size']    = x[4]
            parm_dict['epochs']        = x[5]
            parm_dict['activation']    = x[6]

            results = run(data_dict, parm_dict, count)
            
            save_results(count, parm_dict, results)
            count += 1
    
    job_id += 1
    job.setJob(job_id)
    summary.close()
    
    print('Total time: {:,.0f} minutes'.format((time.time() - start_time)/60))
    sys.exit()