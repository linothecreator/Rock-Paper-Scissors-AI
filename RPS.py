# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import pandas as pd
# from joblib import dump
# from joblib import load
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np

def player(prev_opponent_play, wins=[], losses=[], ties=[], iteration=[], opponent_history=[], player_history=[], train=True):
    num=500
    _span=400
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
      opponent_history=opponent_history[-_span:-1]
      player_history=player_history[-_span:-1]
      wins=wins[-_span:-1]
      move_to_num = {'R': 0, 'P': 1, 'S': 2}
      player_history = [move_to_num[move] for move in player_history]
      opponent_history = [move_to_num[move] for move in opponent_history]
      y = opponent_history
      y = np.array(y).reshape(1, -1)  
      y = np.array([opponent_history]).T
      X = np.array([wins, player_history]).T
      # print(X)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
      model = DecisionTreeClassifier()
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      print(accuracy)
      num_to_move = {0: 'R', 1: 'P',  2: 'S'}
      prediction = [num_to_move[move] for move in y_pred]
      # print(prediction[-1])
      guess = prediction[-1]
      
      
    
    return guess
