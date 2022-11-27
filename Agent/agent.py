import random
import torch
from collections import deque
import pandas as pd
import numpy as np
from Model.model import QNet, QTrainer
learning_rate = 0.001
class Agent:

    def __init__(self, cash):
        
        self.cash = cash
        self.init_balance = cash
        self.holdings_value = 0
        self.value = cash
        self.n_stocks = 0
        self.memory = deque(maxlen=1000000)
        self.holdings_average = 0
        self.performance = 0
        self.model = QNet(6,120,3)

        learning_rate = 0.001
        self.trainer = QTrainer(self.model, learning_rate, 0.9)
        self.epochs = 0

    def getState(self, data):
        state = [data[1]['Open'], data[1]['Predicted Prices'], self.n_stocks, 
        self.cash, self.holdings_value, self.holdings_average, self.performance]
        return state
    
    def getAction(self,state):
        self.epsilon = 300 - self.epochs
        final_action = [0,0,0]
        if random.randint(0, 240) < self.epsilon:
            move = random.randint(0,2)
            final_action[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_action[move] = 1
        return final_action

    def openPosition(self, data):
        if(self.cash >= data[1]['Open'] * 5):
            self.cash -= data[1]['Open'] * 5
            self.holdings_value = self.n_stocks * data[1]['Open'] + data[1]['Open'] * 5
            self.n_stocks += 5
            self.holdings_average = self.holdings_value / self.n_stocks
            self.value = self.cash + self.holdings_value
            return True
        else:
            return False
    def closePosition(self, data):
        if(self.n_stocks >= 5):
            self.cash += data[1]['Open'] * 5
            self.holdings_value = self.n_stocks * data[1]['Open'] - data[1]['Open'] * 5
            self.holdings_average = self.holdings_value / self.n_stocks
            self.n_stocks -= 5
            return True
        else:
            return False
    def calcPerformance(self):
        if(self.value/self.init_balance >= 1):
            self.performance = self.value/self.init_balance
        else:
             self.performance = 1 - self.value/self.init_balance
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def trainLongMemory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)