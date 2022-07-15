# 安裝步驟

1. create .env檔並輸入
    ```
    NAME = new_train_target_algorithm
    GYM = Taxi-v3
    EPISODES = 10000
    WARMUP_EPISODE = 10
    EPSILON = 1
    EPSILON_MIN = 0.5
    EPSILON_DECAY = 0.004
    LR = 1e-3
    LR_MIN = 1e-4
    LR_DECAY = 5000
    GAMMA = 0.99
    STATE_NUM = 500
    EMBEDDING_SIZE = 4
    HIDDEN_SIZE = 50
    ACTION_NUM = 6
    MAX_ACTION = 100
    MAX_QUEUE = 50000
    MAX_EPISODE_STEPS = 5000000
    BATCH_SIZE = 64
    UPDATE_RATE = 20
    LOAD_WEIGHT = new_train_target_algorithm
    ```

2. 下載以下函示庫
    ```bash
    pip3 install tensorflow
    pip3 install python-dotenv 
    pip3 install progressbar 
    pip3 install gym
    pip3 install pygame
    pip3 install matplotlib
    ```

3. 運行程式碼
    ```bash
    python3 main.py
    ```