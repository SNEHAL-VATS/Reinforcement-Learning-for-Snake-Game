import torch
import random 
import numpy as np
from game import SnakeGame, Direction, Point
from collections import deque
from model import Linear_QNet, QNetTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.num_of_games = 0
        self.epsilon = 0 #calculates randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen = MAX_MEMORY) #popleft() if memory exceeds
        self.model = Linear_QNet(11, 256, 3)#total states, hidden states, output must be 3; 
        self.trainer = QNetTrainer(self.model, lr = LEARNING_RATE, gamma = self.gamma)
        #TODO:model, trainer

    def curr_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)
        
        move_left = game.direction == Direction.LEFT
        move_right = game.direction == Direction.RIGHT
        move_up = game.direction == Direction.UP
        move_down = game.direction == Direction.DOWN

        state = [
            #death straight
            (move_right and game.is_collision(point_right)) or
            (move_left and game.is_collision(point_left)) or
            (move_up and game.is_collision(point_up)) or
            (move_down and game.is_collision(point_down)),

            #death right
            (move_up and game.is_collision(point_right)) or
            (move_down and game.is_collision(point_left)) or
            (move_left and game.is_collision(point_up)) or
            (move_right and game.is_collision(point_down)),

            #death left
            (move_down and game.is_collision(point_right)) or
            (move_up and game.is_collision(point_left)) or
            (move_right and game.is_collision(point_up)) or
            (move_left and game.is_collision(point_down)),
            
            #next step to be taken
            move_left,
            move_right,
            move_up,
            move_down,

            #apple/food location
            game.food.x < game.head.x,#turn left
            game.food.x > game.head.x,#turn right
            game.food.y < game.head.y,#turn up
            game.food.y > game.head.y,#turn down

            
        ]

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) #popleft if memory exceeds

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            temp_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            temp_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*temp_sample)
        self.trainer.train_move(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_move(state, action, reward, next_state, game_over)

    def curr_action(self, state):
        #we will statrt with random moves but as the game progresses we will have a set of selected
        #moves to train and then set the snake to for a flawless game.

        #hard coded the randomness
        self.epsilon = 80 - self.num_of_games
        final_step = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            st = random.randint(0,2)
            final_step[st] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            pred = self.model(state0)
            st = torch.argmax(pred).item()
            final_step[st] = 1

        return final_step
def train():
    score_graph = []
    mean_score_graph = []
    total = 0
    temp = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        #get current state
        curr = agent.curr_state(game)

        #get direction
        dir = agent.curr_action(curr)

        #perform action and get new state
        reward, game_over, score = game.play_move(dir)
        stnext = agent.curr_state(game)

        #train short memory
        agent.train_short_memory(curr, dir, reward, stnext, game_over)

        #store in memory
        agent.remember(curr, dir, reward, stnext, game_over)

        if game_over:
            #train the long memory, replay memory
            game.restart()
            agent.num_of_games += 1
            agent.train_long_memory()

            if score > temp:
                temp = score
                agent.model.save()

            print("\nGames played : ",agent.num_of_games,"\tScore : ",score,"\t Current Score : ",temp)

            #graphing the values
            score_graph.append(score)
            total += score
            mean_score = total/agent.num_of_games
            mean_score_graph.append(mean_score)
            plot(score_graph,mean_score_graph)

if __name__ == "__main__":
    train()
