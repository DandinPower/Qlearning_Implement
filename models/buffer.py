from collections import deque
import random

class ReplayBuffer:
    
    #初始化
    def __init__(self, _maxlen):
        self.memory = deque(maxlen= _maxlen)
    
    #儲存一個step的資訊
    def Add(self, _st, _at, _rt, _st1, _done):
        temp = (_st, _at, _rt, _st1, _done)
        self.memory.append(temp)

    #取得batch資料
    def GetBatchData(self, _batchSize):
        data = random.sample(self.memory, _batchSize)
        return data 
    
    #取得目前長度
    def GetLength(self):
        return len(self.memory)

if __name__ == '__main__':
    buffer = ReplayBuffer(200)
    for i in range(10):
        buffer.Add(i, i, i, i, False)
    print(buffer.GetBatchData(20))