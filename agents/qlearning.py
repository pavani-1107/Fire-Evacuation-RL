import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate = 0.1, discount = 0.99, epsilon = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        #QTable: dictionary to store Q-values
        self.q_table = {}

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state][action]
    
    def choose_action(self,state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size) #Explore
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_size)
            return np.argmax(self.q_table[state]) #Exploit
        
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        #Q (s, a) = Q(s, a) + a[r + gamma*max(Q(s',a')) - Q(s, a)]
        current_q = self.get_q_value(state, action)

        if done:
            target_q = reward
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(self.action_size)
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        #Update Q-value
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        self.q_table[state][action] += self.lr * (target_q - current_q)

        #Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay