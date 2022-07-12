import tensorflow as tf 
import numpy as np
import random

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.w = self.add_weight(name='w',
            shape=[input_dim, output_dim], initializer=tf.zeros_initializer())
        self.b = self.add_weight(name='b',
            shape=[output_dim], initializer=tf.zeros_initializer())

    def call(self, inputs):
        matmul = tf.matmul(inputs, self.w)
        bias = matmul + self.b
        return bias

class QModel(tf.keras.Model):
    def __init__(self, _stateNum, _embeddingSize, _actionNum, _hiddenSize):
        super(QModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim = _stateNum, output_dim = _embeddingSize, input_length = 1)
        self.dense1 = LinearLayer(_embeddingSize, _hiddenSize)
        self.dense2 = LinearLayer(_hiddenSize, _hiddenSize)
        self.dense3 = LinearLayer(_hiddenSize, int(_hiddenSize / 2))
        self.action = LinearLayer(int(_hiddenSize / 2), _actionNum)
        self.relu = tf.keras.layers.ReLU()
        
    def call(self, inputs):
        #inputs [state]
        output1 = self.embedding(inputs)
        output2 = self.dense1([output1])
        output3 = self.relu(output2)
        output4 = self.dense2(output3)
        output5 = self.relu(output4)
        output6 = self.dense3(output5)
        output7 = self.relu(output6)
        output8 = self.action(output7)
        return output8 

    def Copy(self, _qModel):
        parameter = _qModel.get_weights()
        self.set_weights(parameter)

    def GetAction(self, st, epsilon):
        if random.uniform(0, 1) <= epsilon:
            return random.randint(0, 5)
        else:
            q_values = self.call(int(st))[0]
            return np.argmax(q_values)

if __name__ == '__main__':
    #q = QModel(500, 6, 50)
    #targetQ = QModel(500, 6, 50)
    #targetQ.Copy(q)
    #print(targetQ(457))
    pass