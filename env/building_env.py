import numpy as np

class BuildingEnv:
    def __init__(self, grid_size=(10,10), fire_spread_prob=0.3):
        self.rows, self.cols = grid_size
        self.fire_spread_prob = fire_spread_prob

        self.EMPTY = 0
        self.WALL = 1
        self.FIRE = 2
        self.EXIT = 3
        self.AGENT = 4

        self.grid = np.zeros((self.rows, self.cols), dtype = int)

        self.agent_pos = None

        self.action_space = 5

    def reset(self):
        """Reset environment to initial state."""
        #Clear grid
        self.grid = np.zeros((self.rows, self.cols), dtype = int)

        #Place agent at bottom-left
        self.agent_pos = [self.rows -1, 0]

        #Place exit at top-right
        self.grid[0, self.cols - 1] = self.EXIT

        #Place initial fire in middle
        self.grid[self.rows // 2, 1] = self.FIRE

        return self._get_state()
    
    def _get_state(self):
        """Return current state as tuple (x, y, fire_neighbors)."""
        x, y = self.agent_pos
        fire_neighbors = self._count_fire_neighbors(x, y)
        return (x, y, fire_neighbors)
    
    def _count_fire_neighbors(self, x, y):
        """Count number of adjacent fire cells."""
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx, ny] ==self.FIRE:
                    count += 1
        return count
    
    def step(self, action):
        """Take action and return next state, reward, done."""

        #Move Agent
        x, y = self.agent_pos

        if action == 0  and x > 0: #Up
            x-= 1

        elif action == 1 and x < self.rows - 1: #Down
            x +=1

        elif action == 2 and y > 0: #Left
            y -= 1

        elif action == 3 and y < self.cols - 1: #Right
            y += 1

        self.agent_pos = [x, y]

        cell = self.grid [x, y]

        if cell == self.EXIT:
            return self._get_state(), 100, True, {} #WIN
        
        elif cell == self.FIRE:
            return self._get_state(), -100, True, {} #LOSE
        
        else:
            self._spread_fire()
            return self._get_state(), -1, False, {} #Continue
        
    def _spread_fire(self):
        new_fires = []
        for i in range (self.rows):
            for j in range (self.cols):
                if self.grid[i, j] == self.FIRE:
                    #check 4 neighbors
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            if self.grid[ni, nj] == self.EMPTY:
                                if np.random.random() < self.fire_spread_prob:
                                    new_fires.append((ni, nj))

        for i, j in new_fires:
            self.grid[i, j] = self.FIRE                            
