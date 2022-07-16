import tensorflow as tf 
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.w = self.add_weight(name = "w",
            shape=[input_dim, output_dim], initializer="random_normal", trainable = True)
        self.b = self.add_weight(name = "b",
            shape=[output_dim], initializer="random_normal", trainable = True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class QModel(tf.keras.Model):
    def __init__(self, _stateNum, _embeddingSize, _actionNum, _hiddenSize):
        super(QModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim = _stateNum, output_dim = _embeddingSize)
        self.dense1 = LinearLayer(_embeddingSize, _hiddenSize)
        self.dense2 = LinearLayer(_hiddenSize, _hiddenSize)
        self.action = LinearLayer(_hiddenSize, _actionNum)
        self.relu = tf.keras.layers.ReLU()
        
    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.expand_dims(x, 0)
        
        x = self.dense1(x)
        x = self.relu(x)
        print(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.action(x)
        x = tf.squeeze(x)
        return x

    def Copy(self, _qModel):
        parameter = _qModel.get_weights()
        self.set_weights(parameter)

    def GetAction(self, st, epsilon):
        if random.uniform(0, 1) <= epsilon:
            return random.randint(0, 5)
        else:
            q_values = self.call(int(st))
            return np.argmax(q_values)

if __name__ == '__main__':

    q = Sequential()
    q.add(Embedding(500, 4, input_length=1))
    q.add(Reshape((4,)))
    q.add(Dense(50, activation='relu'))
    q.add(Dense(50, activation='relu'))
    q.add(Dense(6, activation='linear'))


    '''
    buffer = ReplayBuffer(100)
    for i in range(100):
        buffer.Add(i, i, i, i, False)
    batchData = buffer.GetBatchData(20)
    states = np.array([d[0] for d in batchData])
    actions = np.array([d[1] for d in batchData])
    rewards = np.array([d[2] for d in batchData])
    next_states = np.array([d[3] for d in batchData])
    dones = np.array([d[4] for d in batchData])'''
    q = QModel(500, 4, 6, 50)
    target = QModel(500, 4, 6, 50)
    #print(q.trainable_weights)
    
    #q.save_weights('test.h5')
    q(10)
    q.load_weights(f'weight/good_v2.h5')
    q(100)
    #print(states)
    #print(q.variables)
    #print(q(42))
    #print(q.GetAction(400, 1))
    '''
    lossF = tf.keras.losses.Huber()
    with tf.GradientTape() as tape:
        model_output = q(states)
        target_output = q(next_states)
        model_output = tf.gather_nd(model_output, tf.expand_dims(actions, 1), 1)
        next_state_values = tf.math.reduce_max(target_output, axis = 1)
        expected_q_values = ((1 - dones) * next_state_values * 0.99) + rewards
        loss = lossF(model_output, expected_q_values)    
        grads = tape.gradient(loss, q.variables)
        print(grads)
        #optimizer.apply_gradients(grads_and_vars=zip(grads, self.q.variables))'''