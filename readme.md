# 安裝步驟

1. create .env檔並輸入
    ```
    NAME = test_1
    LOSS_NAME = Huber
    GYM = Taxi-v3
    EPISODES = 10000
    WARMUP_EPISODE = 10
    EPSILON = 1
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 400
    LR = 0.001
    LR_MIN = 0.0001
    LR_DECAY = 5000
    GAMMA = 0.99
    STATE_NUM = 500
    EMBEDDING_SIZE = 4
    HIDDEN_SIZE = 50
    ACTION_NUM = 6
    MAX_ACTION = 100
    MAX_QUEUE = 50000
    MAX_EPISODE_STEPS = 5000000
    BATCH_SIZE = 128
    UPDATE_RATE = 20
    LOAD_WEIGHT = test_1
    ```

2. 下載以下函示庫
    ```bash
    pip3 install tensorflow
    pip3 install python-dotenv 
    pip3 install tqdm
    pip3 install gym
    pip3 install pygame
    pip3 install matplotlib
    ```

3. 運行程式碼
    ```bash
    python3 train.py
    ```

4. 訓練完後運行遊戲
    ```bash
    python3 play.py
    ```