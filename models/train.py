import tensorflow as tf 
import numpy as np 
import random 
import gym
import time
from .q_model import QModel
from .buffer import ReplayBuffer
def Accuracy(y, y_pred):
    y_total = 0
    y_pred_total = 0
    for i in range(len(y)):
        y_total += abs(y[i][0])
        y_pred_total += abs(y_pred[i][0])
    return y_pred_total / y_total

class DeepQlearning:
    
    def __init__(self, _env, _stateNum, _embeddingSize, _actionNum, _hiddenSize, _batchSize):
        self.env = _env
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.001
        self.epsilon = 1
        self.gamma = 0.618
        self.max_action = 100000
        self.batchSize = _batchSize
        self.q = QModel(_stateNum, _embeddingSize, _actionNum, _hiddenSize)
        self.targetQ = QModel(_stateNum, _embeddingSize, _actionNum, _hiddenSize)
        self.buffer = ReplayBuffer(2000)
        self.UpdateTargetNetwork()

    def UpdateTargetNetwork(self):
        self.targetQ.Copy(self.q)
    
    def UpdateEpsilon(self,episode):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min+(1-self.epsilon_min)*np.exp(-self.epsilon_decay*episode) 
    
    def CountBatchTarget(self, X):
        states = np.array([d[0] for d in X])
        next_states = np.array([d[3] for d in X])
        y = self.q(states).numpy()[0]
        q = self.targetQ(next_states).numpy()[0]
        for i, (_, action, reward, _, done) in enumerate(X):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target
        return states, y
    
    def StepTrain(self, X, Y):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) 
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            y_pred = self.q(X)[0]
            loss = mse(y_true=Y, y_pred=y_pred)      
            grads = tape.gradient(loss, self.q.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, self.q.variables))

    def Episode(self, episode):
        st = self.env.reset()
        reward_sum = 0
        action_nums = 0
        cStep = 0
        done = False 
        while not done and action_nums < self.max_action:
            at = self.q.GetAction(st, self.epsilon)
            st1, rt, done, info = self.env.step(at)
            action_nums += 1
            reward_sum += rt
            self.buffer.Add(st, at, rt, st1, done)
            st = st1
            if self.buffer.GetLength() > self.batchSize:
                X = self.buffer.GetBatchData(self.batchSize)
                X,Y = self.CountBatchTarget(X)
                self.StepTrain(X, Y)
                cStep += 1
            if cStep % self.batchSize == 0:
                self.UpdateTargetNetwork()
        print(f'episode:{episode}, reward_sum: {reward_sum}, action_nums: {action_nums}')
    
    def Train(self, _episodeNums):
        for i in range(_episodeNums):
            self.Episode(i)
            self.UpdateEpsilon(i)
            if i%5 == 0:
                print(f'save weight...')
                self.q.save_weights('weight/taxi_model.h5')
    
    def LoadParameter(self):
        self.q(10)
        self.q.load_weights('weight/taxi_model.h5')
    
    def play(self):
        print('start play...')
        observation = self.env.reset()
        count = 0
        reward_sum = 0
        random_episodes = 0
        while random_episodes < 1:
            self.env.render()
            time.sleep(0.5)
            x = observation
            q_values = self.q(int(x))[0]
            print(x, q_values)
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)
            count += 1
            reward_sum += reward
            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()   
        self.env.close()

if __name__ == '__main__':
    '''
    buffer = ReplayBuffer(200)
    for i in range(200):
        buffer.Add(i, 4, i, i, False)
    X = buffer.GetBatchData(20)
    test = DeepQlearning(0, 500, 6, 50, 20)
    X, Y = test.CountBatchTarget(X)
    test.Train(X,Y)'''
    test = DeepQlearning(1, 1, 1, 1, 1)
    print(test.epsilon)
    test.UpdateEpsilon(200)
    print(test.epsilon)
    test.UpdateEpsilon(200)
    print(test.epsilon)


