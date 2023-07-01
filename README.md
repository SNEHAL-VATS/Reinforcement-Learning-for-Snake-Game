# Reinforcement-Learning-for-Snake-Game
Reinforcement Learning with PyTorch

##### I have created this project in a Windows OS. So, I have put the code on how to run it in a Windows OS with virtual environment, follow the steps given below in a cmd

```bash
C:\Users\Folder_of_Choice>pip install python
C:\Users\Folder_of_Choice>curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
C:\Users\Folder_of_Choice>python get-pip.py
C:\Users\Folder_of_Choice>mkdir snake_game
C:\Users\Folder_of_Choice>cd snake_game
C:\Users\Folder_of_Choice\snake_game>virtualenv venv
C:\Users\Folder_of_Choice\snake_game>venv\Scripts\activate
(venv)C:\Users\Folder_of_Choice\snake_game>pip install pygame
C:\Users\Folder_of_Choice\snake_game>pip install torch torchvision
C:\Users\Folder_of_Choice\snake_game>pip install matplotlib ipython
C:\Users\Folder_of_Choice\snake_game>python agent.py
```
## Description
This project is a learning attempt to understand the dyanmics of reinforcement learning with the help of a game we all know and love.

The code will teach the snake to keep making moves as per the rules of the game (eat apple and gain a block and the gameis over if it hits the boundary or itself).

We have a ***game.py*** to create the game and the snake, ***agent.py*** to to create a working agent that will determine the rules, point system and the steps to take
to continue playing and gain a high score. ***Model.py*** will be the file to hold the PyTorch tensors and learning and training code while ***plotter.py*** is just there to plot each game's score and mean square error.

Let the agent run and after 100 games, the snake will begin to play the game with a very small error margin. My laptop is not quite strong enough to let it run for a while, so it will be good if you let it run for an hour or so to see better results.
