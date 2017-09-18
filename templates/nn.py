import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import sys
import time

TB_DIR   = '/home/tom/ML/tb/'                 # where to store TensorBoard data
SAVE_DIR = "/home/tom/ML/tf_checkpoints/"     # where to store the saved model

def run(data, parms, iteration):
    run_id = 'RUN'+str(iteration)
    print("---------------------")
    print('{}'.format(run_id))
    
    # get the true response rate to compare to my predictions
    true_rr = np.sum(data['val_labels'][:,-1]) / data['val_labels'].shape[0]
    
    feature_count = data['train_x'].shape[1]
    num_classes   = np.unique(data['train_labels']).shape[0]
    
    # Load hyper-parameters
    L1         = parms['l1_size']
    LR         = parms['learning_rate']
    LAMBDA     = parms['lambda']
    WEIGHT     = parms['weight']
    BATCH      = parms['batch_size']
    EPOCHS     = parms['epochs']
    ACTIVATION = parms['activation']
        
    # Set up the network
    tf.reset_default_graph()
    x  = tf.placeholder("float", shape=[None, feature_count], name="input")
    y_ = tf.placeholder("float", shape=[None, num_classes])

    l1_w = tf.Variable(tf.truncated_normal([feature_count, L1], dtype=tf.float32))
    l1_b = tf.Variable(tf.truncated_normal([1,L1], dtype=tf.float32))
    
    if   ACTIVATION == 'tanh':
        l1_act = tf.nn.tanh(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'leakyReLU':
        l1_act   = leakyReLU(x, l1_w, l1_b)
    elif ACTIVATION == 'ReLU':
        l1_act   = tf.nn.relu(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'ReLU6':
        l1_act   = tf.nn.relu6(tf.matmul(x,l1_w) + l1_b)
        
    l2_w   = tf.Variable(tf.truncated_normal([L1,num_classes], dtype=tf.float32))
    l2_b   = tf.Variable(tf.truncated_normal([1,num_classes]))
    
    l2_out = tf.add(tf.matmul(l1_act, l2_w), l2_b, name="L2")
    
    # Cost function
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_, logits=l2_out, pos_weight=WEIGHT))
    
    # Optimizer
    optimize = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

    tf.summary.scalar('Cost', cost)
    merged = tf.summary.merge_all()

    # Run
    count = 1
    num_training_batches = int(len(data['train_x']) / BATCH)
    print('{} epochs of {} iterations with batch size {}'.format(EPOCHS,num_training_batches,BATCH))
    
    saver      = tf.train.Saver()
    
    CP = tf.ConfigProto( device_count = {'GPU': 1} )           # set to 0 if you want CPU only
    
    sess = tf.Session(config=CP)
    train_writer = tf.summary.FileWriter(TB_DIR + run_id, sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        a,b = shuffle(data['train_x'],data['train_labels'])
        for j in range(num_training_batches):
            x_mini = a[j*BATCH:j*BATCH+BATCH]
            y_mini = b[j*BATCH:j*BATCH+BATCH]
            _ = sess.run([optimize], feed_dict = {x: x_mini, y_: y_mini})
            if j % 20 == 0:
                c = sess.run(merged, feed_dict = {x: data['val_x'], y_: data['val_labels']})
                train_writer.add_summary(c, count)
                count += 1
                
        # At the end of each epoch count the number of True Positives in a given batch (batch size is K)
        L2 = sess.run(l2_out, feed_dict = {x: data['val_x'], y_: data['val_labels']})
        K = 100000
        ix = np.argpartition(L2[:,1], -K)[-K:]
        
        topK  = data['val_labels'][ix][:,1].sum()
        my_rr = topK / K
        
        lift = my_rr / true_rr-1
        print("Epoch {} True Positives: {:.0f} with lift: {:.2%}".format(i,topK,lift))
        
    '''L1W = l1_w.eval(session=sess)
    L1B = l1_b.eval(session=sess)
    L2W = l2_w.eval(session=sess)
    L2B = l2_b.eval(session=sess)
    np.savetxt("/home/tom/L1W"+run_id+".csv", L1W, delimiter=",")
    np.savetxt("/home/tom/L1B"+run_id+".csv", L1B, delimiter=",")
    np.savetxt("/home/tom/L2W"+run_id+".csv", L2W, delimiter=",")
    np.savetxt("/home/tom/L2B"+run_id+".csv", L2B, delimiter=",")'''
    
    saver.save(sess, SAVE_DIR+'SS_'+run_id )
    train_writer.close()
    return (topK, lift)
