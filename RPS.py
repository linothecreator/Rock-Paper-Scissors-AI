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


def player(prev_opponent_play,
           wins=[],
           losses=[],
           ties=[],
           iteration=[],
           opponent_history=[],
           player_history=[],
           counter=[0],
           train=True,
           play_order=[{
             "RR": 0,
             "RP": 0,
             "RS": 0,
             "PR": 0,
             "PP": 0,
             "PS": 0,
             "SR": 0,
             "SP": 0,
             "SS": 0,
           }]):
  num = 5
  _span = 199
  if prev_opponent_play == '':
    prev_opponent_play = 'S'

  counter[0] += 1
  choices = ["R", "P", "S", "P", "S"]
  guess = choices[counter[0] % len(choices)]

  _w = wins.count('1')
  _tot = wins.count('1') + losses.count('1')
  if _tot !=0:
    win_rate=_w/_tot
    print('my_win_rate', win_rate)

  print('player: ', player_history)
  print('opponent: ', opponent_history)


  # Counter ABBEY
  #======================================================================
  def counter_abbey():
    last_two = "".join(player_history[-2:])
    prev_play = player_history[-1]
    # Counter for last opp move pair
    if len(last_two) == 2:
      play_order[0][last_two] += 1
# 3 potential opponent move pairs
    potential_plays = [
      prev_play + "R",
      prev_play + "P",
      prev_play + "S",
    ]
    # Check the play order for the 3 potential plays
    sub_order = {
      k: play_order[0][k]
      for k in potential_plays if k in play_order[0]
    }
    # check for more frequent of 3 potential plays
    prediction = max(sub_order, key=sub_order.get)[-1:]
    ideal_response = {'P': 'R', 'R': 'S', 'S': 'P'}
    guess = ideal_response[prediction]
    return guess

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

  def getresults(player_history, opponent_history):
    outcome = getoutcome(player_history[-2], opponent_history[-1])
    print(player_history[-2], opponent_history[-1])
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
  # iteration.append(len(player_history))
  if len(player_history) >= 2:
    getresults(player_history, opponent_history)

  if len(opponent_history) >= num:
    opponent_history = opponent_history[-_span:-1]
    player_history = player_history[-_span:-1]
    wins = wins[-_span:-1]
    move_to_num = {'R': 0, 'P': 1, 'S': 2}
    player_history = [move_to_num[move] for move in player_history]
    opponent_history = [move_to_num[move] for move in opponent_history]
    y = opponent_history
    y = np.array(y).reshape(1, -1)
    y = np.array([opponent_history]).T
    X = np.array([wins, player_history]).T
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=41)
    model = DecisionTreeClassifier()
    # model.fit(X_train, y_train)
    model.fit(X, y)
    # y_pred = model.predict(X_test)
    y_pred = model.predict(X)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy)
    num_to_move = {0: 'R', 1: 'P', 2: 'S'}
    prediction = [num_to_move[move] for move in y_pred]
    # print(prediction[-1])
    prediction = prediction[-1]

    #beats quincy
    # ideal_response = {'P': 'R', 'R': 'S', 'S': 'P'}

    #beats kris
    # ideal_response = {'P': 'P', 'R': 'R', 'S': 'S'}

    #beats mruguesh
    # ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

    #abbey
    ideal_response = {'P': 'P', 'R': 'R', 'S': 'S'}

    prediction = ideal_response[prediction]

    guess = prediction

  return guess
