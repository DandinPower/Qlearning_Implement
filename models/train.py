from progressbar import ProgressBar
from .q_model import QModel
from .buffer import ReplayBuffer
from .draw import History
from .config import Config
import tensorflow as tf 
from tensorflow.keras import backend as K
import numpy as np 
import random 
import gym
import time

class DeepQlearning:
    #初始化參數及物件
    def __init__(self, _config):
        self.config = _config
        self.episodes = _config.episodes
        self.env = gym.make(_config.gym)
        self.env._max_episode_steps = _config.max_episode_steps
        self.epsilon = _config.epsilon
        self.epsilon_max = self.epsilon
        self.epsilon_min = _config.epsilon_min
        self.epsilon_decay = _config.epsilon_decay
        self.gamma = _config.gamma
        self.max_action = _config.max_action
        self.updateRate = _config.updateRate
        self.lr = _config.lr
        self.lr_min = _config.lr_min
        self.lr_max = _config.lr
        self.lr_decay = _config.lr_decay
        self.max_queue = _config.max_queue
        self.batchSize = _config.batchSize
        #self.q = QModel(_config.stateNum, _config.embeddingSize, _config.actionNum, _config.hiddenSize)
        #self.targetQ = QModel(_config.stateNum, _config.embeddingSize, _config.actionNum, _config.hiddenSize)
        self.q = self.GetQModel()
        self.targetQ = self.GetQModel()
        self.buffer = ReplayBuffer(self.max_queue)
        self.history = History()
        self.loss = self.GetLossFunction()
        self.UpdateTargetNetwork()

    def GetQModel(self):
        model = Sequential()
        moedl.add(Embedding(500, 4, input_length=1))
        model.add(Reshape((4,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(6, activation='linear'))
        return model

    #根據config來決定loss function
    def GetLossFunction(self):
        if self.config.lossName == "Mse":
            return tf.keras.losses.MeanSquaredError()
        elif self.config.lossName == "Huber":
            return tf.keras.losses.Huber()
        else:
            return tf.keras.losses.Huber()

    #將model複製給target model
    def UpdateTargetNetwork(self):
        self.targetQ(0)
        self.q(0)
        self.targetQ.Copy(self.q)
    
    #根據目前的episode調整leaning_rate
    def UpdateLearningRate(self, episode):
        delta = self.lr_max - self.lr_min
        base = self.lr_min
        rate = self.lr_decay
        self.lr = base + delta * np.exp(-episode / rate)

    ##根據目前的episode調整epsilon
    def UpdateEpsilon(self,episode):
        delta = self.epsilon_max - self.epsilon_min
        base = self.epsilon_min
        self.epsilon = base + delta * np.exp(-episode / self.epsilon_decay)
    
    def Optimize(self, batchData):
        states = np.array([d[0] for d in batchData])
        actions = np.array([d[1] for d in batchData])
        rewards = np.array([d[2] for d in batchData])
        next_states = np.array([d[3] for d in batchData])
        dones = np.array([d[4] for d in batchData])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr) 
        with tf.GradientTape() as tape:
            model_output = self.q(states)
            target_output = self.targetQ(next_states)
            model_output = tf.gather_nd(model_output, tf.expand_dims(actions, 1), 1)
            next_state_values = tf.math.reduce_max(target_output, axis = 1)
            expected_q_values = ((1 - dones) * next_state_values * 0.99) + rewards
            loss = self.loss(expected_q_values, model_output)    
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
            if episode > self.config.warm_up:
                self.UpdateLearningRate(episode - self.config.warm_up + 1)
            if self.buffer.GetLength() > self.batchSize:
                X = self.buffer.GetBatchData(self.batchSize)
                self.Optimize(X)
                cStep += 1
            if cStep % self.updateRate == 0:
                self.UpdateTargetNetwork()
        self.history.AddHistory([episode, reward_sum, action_nums, self.epsilon])
    
    def LoadParameter(self):
        self.q(10)
        self.q.load_weights(f'weight/{self.config.loadName}.h5')
    
    def Train(self):
        startTime = time.time()
        j = 0
        total = self.episodes
        pBar = ProgressBar().start()
        for i in range(total):
            self.Episode(i)
            self.UpdateEpsilon(i)
            if i%999 == 0:
                self.q.save_weights(f'weight/{self.config.name}.h5')
            pBar.update(int((j / (total - 1)) * 100))
            j += 1
        pBar.finish()
        print(f'cost time: {round(time.time() - startTime,3)} sec')
        self.history.ShowHistory(f'figure/{self.config.name}.png')
        
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
            q_values = self.q(int(x))
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
    config = Config()
    dpq = DeepQlearning(config)
    for i in range(150):
        dpq.buffer.Add(1,1,1,1,False)
    batchData = dpq.buffer.GetBatchData(64)
    dpq.Optimize(batchData)



