# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd
# from joblib import dump
# from joblib import load
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np

def player(prev_opponent_play, wins=[], losses=[], ties=[], iteration=[], opponent_history=[], player_history=[], train=True):
    num = 100
    if prev_opponent_play == '':
        prev_opponent_play='P'
      
    guess = random.choice(['R', 'P', 'S'])

#Determine Wins
    def getoutcome(playermove, opponent_move):
            if playermove == opponent_move:
              return "tie"
            elif playermove == "R" and opponent_move == "S" or \
               playermove == "P" and opponent_move == "R" or \
               playermove == "S" and opponent_move == "P":
              return "win"
            else:
              return "loss"
# Record Result  
    def getresults(playerhistory, opponent_history):
                  outcome = getoutcome(playerhistory[-1], opponent_history[-1])
                  if outcome == "win":
                      wins.append("1")
                      losses.append("0")
                      ties.append("0")
                  elif outcome == "loss":
                      wins.append("0")
                      losses.append("1")
                      ties.append("0")                    
                  else:
                      wins.append("0")
                      losses.append("0")
                      ties.append("1")   
    
    
    # Update State
    player_history.append(guess)    
    opponent_history.append(prev_opponent_play)
    getresults(player_history, opponent_history)
    iteration.append(len(player_history))

    if len(opponent_history)>=num:
      
      move_to_num = {'R': 0, 'P': 1, 'S': 2}
      player_history = [move_to_num[move] for move in player_history]
      y = opponent_history
      # print(y)
      y = [move_to_num[move] for move in y]
      y = np.array(y).reshape(1, -1)   
      y = y[-1]
      X = np.array([wins, player_history]).T
      # print(X)
      model = DecisionTreeClassifier()
      model.fit(X, y)
      prediction = model.predict(X)
      num_to_move = {0: 'R', 1: 'P',  2: 'S'}
      prediction = [num_to_move[move] for move in prediction]
      print(prediction[-1])
      guess = prediction[-1]
      
      
    
    return guess
