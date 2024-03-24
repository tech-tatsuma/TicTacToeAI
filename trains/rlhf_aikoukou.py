import torch
import numpy as np
from collections import namedtuple

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from games.tictactoe import initialize_board, print_board, player_move, check_winner
from trains.train_utils import select_action, optimize_model, ReplayMemory, select_action_rlhf
from models.dqn import DQN

STATE_SAVE_PATH = 'training_state_koukou.json'

def save_training_state(steps_done, episode):
    """
    Saves the progress of training to an external file.
    """
    state = {
        'steps_done': steps_done,
        'episode': episode,
    }
    with open(STATE_SAVE_PATH, 'w') as f:
        json.dump(state, f)

def load_training_state():
    """
    Loads the training progress from an external file.
    """
    if os.path.exists(STATE_SAVE_PATH):
        with open(STATE_SAVE_PATH, 'r') as f:
            state = json.load(f)
        return state['steps_done'], state['episode']
    else:
        return 0, 0 

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
    row, col = divmod(action.item(), 3)

    symbols = ['X', 'O']
    symbol = symbols[player - 1]
    board[row][col] = symbol

    next_state = np.array(board)

    winner = check_winner(next_state)

    done = winner != -1

    reward = 0
    if winner == player:
        reward = 1
    elif winner == 0:
        reward = 0.5  
    elif winner == 3 - player:
        reward = -1 

    return next_state, reward, done

def evaluate_board(board, player):
    """
    Evaluates the board to return if the game has ended and what the reward is.
    """
    winner = check_winner(board)
    done = winner != -1
    if winner == player:
        reward = 1
    elif winner == 0:
        reward = 0.5
    else:
        reward = 0  # 引き分けまたはゲームが続く場合
    return done, reward

def rlhf(num_episodes=100):
    """
    Main function to train an agent using DQN.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN().to(device)
    policy_net.load_state_dict(torch.load('policy_net_2_final_weights.pt', map_location=device))
    target_net = DQN().to(device)
    target_net.load_state_dict(torch.load('target_net_2_final_weights.pt', map_location=device))
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    memory = ReplayMemory(MEMORY_SIZE)

    steps_done, start_episode = load_training_state()

    BATCH_SIZE = 32
    GAMMA = 0.99
    TARGET_UPDATE = 5
    steps_done = 0

    for episode in range(start_episode, num_episodes):

        board = initialize_board()

        current_player = 1

        state = to_tensor(board)

        done = False

        while not done:

            print_board(board)

            if current_player == 1:
                player_move(board, current_player)
                next_board = np.array(board)
                done, reward = evaluate_board(next_board, current_player)
            else:
                print("AI's turn")

                symbol = -1

                state = to_tensor(board).to(device)

                action = select_action_rlhf(state, policy_net, device, symbol)

                next_board, reward, done = play_step(board, current_player, action)

                next_state = to_tensor(next_board)

                memory.push(state, action, next_state, torch.tensor([reward], device=device))

                state = next_state
                board = next_board

            steps_done += 1

            optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device)
            torch.save(policy_net.state_dict(), 'policy_net_2_final_weights_rlhf.pt')

            current_player = 2 if current_player == 1 else 1

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), 'target_net_2_final_weights_rlhf.pt')

        print_board(board)

        print(f"Episode {episode} complete")

        save_training_state(steps_done, episode + 1)

    print("Training complete")

if __name__ == "__main__":
    rlhf()