from Agent.agent import Agent
import pandas as pd
import matplotlib.pyplot as plt
class Game:
    def __init__(self, data, iterations):
        self.data = data
        self.iterations = iterations
        self.agent = Agent(10000)
        record = 1
        self.df = pd.DataFrame()
        
    def run_test(self):
        n_list = []
        p_list = []
        for n in range(0,self.iterations+1):
            
            for s in self.data.iterrows():
                state = self.agent.getState(s)
                action = self.agent.getAction(state)
                if(action.index(max(action)) == 0):
                    self.agent.openPosition(s)
                elif(action.index(max(action)) == 1):
                    self.agent.closePosition(s)
                else:
                    pass
                self.agent.calcPerformance()
                new_state = self.agent.getState(s)
                self.agent.remember(state, action, 0, new_state, False)
                self.agent.train_short_memory(state, action, 0, new_state, False)
            perfromance = self.agent.performance
            if(perfromance > record):
                record = perfromance
                self.agent.model.save()
                self.agent.remember(state, action, perfromance, new_state, True)
            else:
                self.agent.remember(state, action, 0, new_state, True)
            self.agent.trainLongMemory()
            n_list.append(n)
            p_list.append(perfromance)
            self.agent.epochs += 1
            self.agent.cash = 10000
            self.agent.holdings_value = 0
            self.agent.n_stocks = 0
            self.agent.holdings_average = 0
            self.agent.value = 10000
            self.agent.performance = 1
        self.df['iterations'] = n_list
        self.df['Performance'] = p_list
        plt.plot(self.df['iterations'], self.df['Performance'])