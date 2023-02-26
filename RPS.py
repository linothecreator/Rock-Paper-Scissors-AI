# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import pandas as pd
import numpy as np
from joblib import dump

def player(prev_opponent_play, wins=[], losses=[], ties=[], iteration=[], opponent_history=[], player_history=[], train=True):

    if prev_opponent_play == '':
        prev_opponent_play='P'


#Determine Wins
    def get_outcome(player_move, opponent_move):
            if player_move == opponent_move:
              return "tie"
            elif player_move == "R" and opponent_move == "S" or \
               player_move == "P" and opponent_move == "R" or \
               player_move == "S" and opponent_move == "P":
              return "win"
            else:
              return "loss"
# Record Result  
    def get_results(player_history, opponent_history):
                  outcome = get_outcome(player_history[-1], opponent_history[-1])
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
    
    if train or len(player_history) < 99999:
            guess = random.choice(['R', 'P', 'S'])
    
# Update State
    player_history.append(guess)    
    opponent_history.append(prev_opponent_play)
    get_results(player_history, opponent_history)
    iteration.append(len(player_history))

# Graph Results
    if len(opponent_history)==99999:        

# create pandas dataframe  
        data_state = {'Opponent History': opponent_history,
          'Player History': player_history,
          'Wins': wins, 'Iteration':iteration}      
        df = pd.DataFrame(data_state)
        le = LabelEncoder()
#Encoding the data_state
        df['Opponent History'] = le.fit_transform(df['Opponent History'])
        df['Player History'] = le.fit_transform(df['Player History'])
        df['Wins'] = le.fit_transform(df['Wins'])
        df['Iteration']=le.fit_transform(df['Iteration'])
        data=df        
        print(df)
        df = pd.DataFrame(data)
      
# Plot 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['Opponent History'], df['Wins'], df['Iteration'], c=df['Wins'], cmap='jet')
        
# Add labels and title
        ax.set_xlabel('Opponent History')
        ax.set_ylabel('Wins')
        ax.set_zlabel('Iteration')
        ax.set_title('Wins vs Player and Opponent History')

        X = data[['Opponent History', 'Iteration']] # X is a 2D array
        Y = data['Wins'] # Y is a 1D array
        Z = data['Player History'] # Z is a 1D array

        # Create a 3D array to hold the input data
        # The first dimension corresponds to the rows of the input data
        # The second dimension corresponds to the columns of the input data
        # The third dimension corresponds to the variables (X, Y, Z)
        input_data = np.zeros((len(data), 3))
        input_data[:,0] = X['Opponent History']
        input_data[:,1] = X['Iteration']
        input_data[:,2] = Z
        
        # Split the data into training and testing sets

        X_train, X_test, y_train, y_test = train_test_split(input_data, Y, test_size=0.2, random_state=42)
        
        # Create an MLPClassifier with two hidden layers, each with 100 neurons
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
        
        # Train the model on the training data
        clf.fit(X_train, y_train)
        
        # Use the model to make predictions on the testing data
        y_pred = clf.predict(X_test)
        
        # Evaluate the accuracy of the model

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        dump(clf, 'my_model.joblib')

        
      

    # Predict the opponent's next move using the trained model
    # if len(opponent_history)>100:
        
    
  
    return guess

