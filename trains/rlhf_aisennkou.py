import torch
import numpy as np
from collections import namedtuple

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from games.tictactoe import initialize_board, print_board, player_move, check_winner
from trains.train_utils import optimize_model, ReplayMemory, select_action_rlhf
from models.dqn import DQN

# Filename to save/load the training state
STATE_SAVE_PATH = 'training_state_sennkou.json'

def save_training_state(steps_done, episode):
    """
    Saves the progress of training to an external file.
    """
    state = {
        'steps_done': steps_done, # Number of steps completed
        'episode': episode, # Current episode
    }
    with open(STATE_SAVE_PATH, 'w') as f: # Open the file in write mode
        json.dump(state, f) # Write the state dictionary to the file as JSON

def load_training_state():
    """
    Loads the training progress from an external file.
    """
    if os.path.exists(STATE_SAVE_PATH): # Check if the file exists
        with open(STATE_SAVE_PATH, 'r') as f: # Open the file in read mode
            state = json.load(f) # Load the JSON data from the file
        return state['steps_done'], state['episode'] # Return the loaded steps and episode
    else:
        return 0, 0  # If the file does not exist, return initial values

# Variable initialization
MEMORY_SIZE = 64 # Size of the replay buffer

# Define a transition tuple for use in the replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def to_tensor(board):
    """
    Converts a board state to a tensor. Maps ' ' to 0, 'X' to 1, and 'O' to -1.
    """
    state = [0 if cell == ' ' else 1 if cell == 'X' else -1 for row in board for cell in row]
    return torch.tensor(state, dtype=torch.float).unsqueeze(0)

def play_step(board, player, action):
    """
    Applies a selected action to the board and returns the next state, reward, and a game over flag.
    
    Args:
        board: The current state of the board.
        player: The current player (either 1 or 2).
        action: The action taken (the position of the chosen cell).

    Returns:
        next_state: The state of the board after the action.
        reward: The reward for that action.
        done: A flag indicating if the game has ended.
    """
    # Calculate row and column from the action (cell position)
    row, col = divmod(action.item(), 3)
    # Place the player's symbol on the board
    symbols = ['X', 'O']
    symbol = symbols[player - 1]
    board[row][col] = symbol

    # Convert the board to a NumPy array
    next_state = np.array(board)

    # Determine the winner
    winner = check_winner(next_state)

    # Set the game over flag (the game ends if there is a winner or a draw)
    done = winner != -1
    # Calculate the reward
    reward = 0
    if winner == player:
        reward = 1 # Reward for winning
    elif winner == 0:
        reward = 0.5  # Reward for a draw
    elif winner == 3 - player:
        reward = -1 # Penalty for losing

    return next_state, reward, done

def evaluate_board(board, player):
    """
    Evaluates the board to return if the game has ended and what the reward is.
    """
    winner = check_winner(board)
    done = winner != -1 # Game is over if there is a winner or a draw
    if winner == player:
        reward = 1 # Winning reward
    elif winner == 0:
        reward = 0.5 # Reward for a draw
    else:
        reward = 0  # No reward for a draw or if the game continues
    return done, reward

# Main function to train an agent using Deep Q-Network (DQN)
def rlhf(num_episodes=100):
    """
    Main function to train an agent using DQN.
    """

    # Set up the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the models
    policy_net = DQN().to(device)
    policy_net.load_state_dict(torch.load('policy_net_1_final_weights_rlhf.pt', map_location=device))
    target_net = DQN().to(device)
    target_net.load_state_dict(torch.load('policy_net_1_final_weights_rlhf.pt', map_location=device))
    target_net.eval() # Set the target network to evaluation mode

    # Initialize the optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    # Initialize the replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # Load saved training state
    steps_done, start_episode = load_training_state()

    # Training parameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    TARGET_UPDATE = 10

    # Reset steps done for each training session
    steps_done = 0

    # Training loop
    for episode in range(start_episode, num_episodes):

        # Initialize the game board
        board = initialize_board()
        # Set the current player
        current_player = 1
        # Convert the initial board to tensor
        state = to_tensor(board)
        # Flag to indicate the end of the game
        done = False

        while not done:

            print_board(board)

            # Let the agent select an action
            if current_player == 1:
                print("AI's turn")
                # Set the symbol for AI
                symbol = 1
                # Convert the board to state and transfer to device
                state = to_tensor(board).to(device)
                # Select an action
                action = select_action_rlhf(state, policy_net, device, symbol)
                # Apply the action and get the next state, reward, and game over flag
                next_board, reward, done = play_step(board, current_player, action)
                # Convert the next board to tensor
                next_state = to_tensor(next_board)
                # Add experience to memory
                memory.push(state, action, next_state, torch.tensor([reward], device=device))
                # Update
                state = next_state
                board = next_board
            else:
                # Accept input from human player
                player_move(board, current_player)
                next_board = np.array(board)
                done, reward = evaluate_board(next_board, current_player)

            # Update progress flag
            steps_done += 1

            # Optimize the model
            optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device)
            # Save the policy network's weights
            torch.save(policy_net.state_dict(), 'policy_net_1_final_weights_rlhf.pt')

            # Switch turn
            current_player = 2 if current_player == 1 else 1

        # Update the target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), 'target_net_1_final_weights_rlhf.pt')

        print_board(board)

        print(f"Episode {episode} complete")

        # Save training state after each episode
        save_training_state(steps_done, episode + 1)

    print("Training complete")

if __name__ == "__main__":
    rlhf()