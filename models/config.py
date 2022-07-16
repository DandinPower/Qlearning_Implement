from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    def __init__(self):
        self.name = os.getenv("NAME")
        self.lossName = os.getenv("LOSS_NAME")
        self.episodes = int(os.getenv("EPISODES"))
        self.warm_up = int(os.getenv("WARMUP_EPISODE"))
        self.loadName = os.getenv("LOAD_WEIGHT")
        self.gym = os.getenv("GYM")
        self.epsilon = float(os.getenv("EPSILON"))
        self.epsilon_min = float(os.getenv("EPSILON_MIN"))
        self.epsilon_decay = float(os.getenv("EPSILON_DECAY"))
        self.lr = float(os.getenv("LR"))
        self.lr_min = float(os.getenv("LR_MIN"))
        self.lr_decay = float(os.getenv("LR_DECAY"))
        self.gamma = float(os.getenv("GAMMA"))
        self.stateNum = int(os.getenv("STATE_NUM"))
        self.embeddingSize = int(os.getenv("EMBEDDING_SIZE"))
        self.hiddenSize = int(os.getenv("HIDDEN_SIZE"))
        self.actionNum = int(os.getenv("ACTION_NUM"))
        self.max_action = int(os.getenv("MAX_ACTION"))
        self.max_episode_steps = int(os.getenv("MAX_EPISODE_STEPS"))
        self.max_queue = int(os.getenv("MAX_QUEUE"))
        self.batchSize = int(os.getenv("BATCH_SIZE"))
        self.updateRate = int(os.getenv("UPDATE_RATE"))

        