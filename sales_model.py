
# coding: utf-8

# In[1]:

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[2]:

training_data_df= pd.read_csv("sales_data_training.csv", dtype=float)
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values


# In[3]:

test_data_df= pd.read_csv("sales_data_test.csv", dtype=float)
X_testing = test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values


# In[4]:

X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))


# In[5]:

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)


# In[6]:

X_scaled_test = X_scaler.fit_transform(X_testing)
Y_scaled_test = Y_scaler.fit_transform(Y_testing)


# In[7]:

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))


# In[8]:

RUN_NAME = "run 1 with 50 nodes"
learning_rate = 0.001
training_epochs = 100
display_step = 5


# In[9]:

number_of_inputs = 9
number_of_outputs = 1


# In[10]:

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50


# In[11]:

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))


# In[12]:

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], 
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("baises1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)


# In[13]:

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], 
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)


# In[14]:

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)


# In[15]:

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases


# In[16]:

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))    


# In[17]:

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[21]:

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()


# In[23]:

saver = tf.train.Saver()


# In[25]:

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    training_writer = tf.summary.FileWriter('./logs/{}/training'.format(RUN_NAME),session.graph)
    testing_writer = tf.summary.FileWriter('./logs/{}/testing'.format(RUN_NAME),session.graph)
    
    for epoch in range(training_epochs):
        session.run(optimizer,feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost,summary], feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_summary = session.run([cost,summary], feed_dict={X: X_scaled_test, Y: Y_scaled_test})
            
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            
            print (epoch, training_cost, testing_cost)
    print("Training is complete!")
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_test, Y: Y_scaled_test})
    
    save_path = saver.save(session, "./logs/trained_models.ckpt")
    
    print ("final_training_cost:{}, final_testing_cost:{}".format(final_training_cost, final_testing_cost))


# In[ ]:




# In[ ]:




# In[ ]:



