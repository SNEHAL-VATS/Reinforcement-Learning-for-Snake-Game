import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = func.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QNetTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criteria = nn.MSELoss()
    
    def train_move(self, state, action, reward, stnext, game_over):
        state = torch.tensor(state, dtype=torch.float)
        stnext =  torch.tensor(stnext, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float) 
        #(n,x)
        if len(state.shape) == 1:
            #only one dimension
            #reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            stnext = torch.unsqueeze(stnext, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        #1. predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(game_over)):
            Q_new = reward[i]
            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(stnext[i]))
            target[i][torch.argmax(action[i]).item()] = Q_new    

        #2.Q_new = r + y * max(next_predicted Q value) -> only do this if not game over
        #pred.clone()
        #preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criteria(target, pred)
        loss.backward()

        self.optimizer.step()