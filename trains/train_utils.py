import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
from models.dqn import DQN
import numpy as np

# Setting the training parameters
GAMMA = 0.99  # Discount factor. Determines how much future rewards are valued.
EPS_START = 0.9  # Starting probability for exploration. Initially, the probability of choosing random actions is high.
EPS_END = 0.05  # Minimum probability for exploration. Over time, reduces the probability of random actions.
EPS_DECAY = 200  # Rate of decay for exploration rate. A higher value means a longer exploration period.
TARGET_UPDATE = 10  # Interval for updating the target network (in episodes).
MEMORY_SIZE = 15000  # Size of the replay buffer. Determines how many past experiences are retained.
BATCH_SIZE = 128  # Batch size. The number of data points to train on at once.
LR = 0.001  # Learning rate.

# Defining a transition structure for the replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def find_winning_move(board, ai_symbol):
    """
    Identifies winning moves for the AI, assuming the board is a numeric numpy.ndarray.
    """
    # Loop three times (since the board is 3x3)
    for i in range(3):
        # Extract column data
        row = board[i, :]
        if np.count_nonzero(row == ai_symbol) == 2 and np.count_nonzero(row == 0) == 1:
            return i, np.where(row == 0)[0][0]  # Return the position of the last empty cell for a win
        col = board[:, i]
        if np.count_nonzero(col == ai_symbol) == 2 and np.count_nonzero(col == 0) == 1:
            return np.where(col == 0)[0][0], i  # Return the position of the last empty cell for a win

    # Check diagonals
    diag1 = board.diagonal()
    if np.count_nonzero(diag1 == ai_symbol) == 2 and np.count_nonzero(diag1 == 0) == 1:
        return np.where(diag1 == 0)[0][0], np.where(diag1 == 0)[0][0]  # Return the position of the last empty cell for a win

    diag2 = np.fliplr(board).diagonal()
    if np.count_nonzero(diag2 == ai_symbol) == 2 and np.count_nonzero(diag2 == 0) == 1:
        return np.where(diag2 == 0)[0][0], 2 - np.where(diag2 == 0)[0][0]  # Return the position of the last empty cell for a win

    return None

def prevent_win(board, ai_symbol):
    """
    Identifies actions to prevent the opponent from winning.
    """
    # Determine the enemy player's symbol
    enemy_symbol = -ai_symbol

    for i in range(3):
        row = board[i, :]
        if np.count_nonzero(row == enemy_symbol) == 2 and np.count_nonzero(row == 0) == 1:
            return i, np.where(row == 0)[0][0]  # Return the position of the last empty cell for a win
        col = board[:, i]
        if np.count_nonzero(col == enemy_symbol) == 2 and np.count_nonzero(col == 0) == 1:
            return np.where(col == 0)[0][0], i  # Return the position of the last empty cell for a win

    # Similar checks for rows, columns, and diagonals as in find_winning_move
    # but focused on blocking the opponent's winning move.
    diag1 = board.diagonal()
    if np.count_nonzero(diag1 == enemy_symbol) == 2 and np.count_nonzero(diag1 == 0) == 1:
        return np.where(diag1 == 0)[0][0], np.where(diag1 == 0)[0][0] 

    diag2 = np.fliplr(board).diagonal()
    if np.count_nonzero(diag2 == enemy_symbol) == 2 and np.count_nonzero(diag2 == 0) == 1:
        return np.where(diag2 == 0)[0][0], 2 - np.where(diag2 == 0)[0][0]

    return None

class ReplayMemory(object):
    """A class for the replay buffer."""

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity) # Create a deque with the specified size

    def push(self, *args):
        """Saves a transition to memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly samples a batch from memory and returns it."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the length of the memory (i.e., the number of stored transitions)."""
        return len(self.memory)

def optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device):
    """A function to optimize learning; updates the network weights."""
    # Exit the function early if there aren't enough transitions in memory
    if len(memory) < BATCH_SIZE:
        return
    # Set the policy network to training mode
    policy_net.train()
    
    # Sample a batch of transitions from the replay memory
    transitions = memory.sample(BATCH_SIZE)

    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements (non-final states are not the end of the episode)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    # Concatenate the state, action, and reward batch elements
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Move the policy network to the specified device
    policy_net = policy_net.to(device)

    # Compute Q values for current states (policy_net) using the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the older target_net; selecting their best reward with max(1)[0].
    # This is merged with zeros for final states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss between current Q values and the target Q values
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad() # Reset the gradients to zero
    loss.backward() # Perform backpropagation to compute gradients
    optimizer.step() # Update the weights
    print('optimized')


def select_action(state, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY, device, symbol):
    """
    Selects an action based on the current state using the policy network.
    Utilizes epsilon-greedy strategy for exploration and exploitation.
    """

    # Convert the state to a 3x3 board format
    board = state.view(3, 3).cpu().numpy()

    # Check if there's a winning move available
    winning_move = find_winning_move(board, symbol)
    if winning_move is not None:
        # If there is, return the action leading to a win
        action_index = winning_move[0] * 3 + winning_move[1]
        return torch.tensor([[action_index]], device=device, dtype=torch.long)

    # Check if there's a move to prevent the opponent's win
    action = prevent_win(board, symbol)
    if action is not None:
        # If there is, return the action that blocks the opponent's win
        action_index = action[0] * 3 + action[1] 
        return torch.tensor([[action_index]], device=device, dtype=torch.long)
    
    # Generate a random number for epsilon-greedy selection
    sample = random.random()

    # Calculate the epsilon threshold for exploration
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

    with torch.no_grad():
        q_values = policy_net(state)

    # Get available actions (empty cells)
    available_actions = [i for i in range(9) if state.flatten()[i] == 0]
    # Extract Q-values for available actions only
    q_values_available = q_values[0, available_actions]

    # If the random number is above the threshold, select the best action based on the policy network
    if sample > eps_threshold:
        # Select the action with the highest Q-value
        action = available_actions[torch.argmax(q_values_available).item()]
    else:
        # Otherwise, select a random action from available actions
        action = random.choice(available_actions)

    # Return the selected action
    return torch.tensor([[action]], device=device, dtype=torch.long)

def select_action_rlhf(state, policy_net, device, symbol, epsilon=0.5):

    # Convert the state to a 3x3 board format
    board = state.view(3, 3).cpu().numpy()

    # Check if there's a winning move available
    winning_move = find_winning_move(board, symbol)
    if winning_move is not None:
        # If there is, return the action leading to a win
        action_index = winning_move[0] * 3 + winning_move[1]
        return torch.tensor([[action_index]], device=device, dtype=torch.long)

    # Check if there's a move to prevent the opponent's win
    action = prevent_win(board, symbol)
    if action is not None:
        # If there is, return the action that blocks the opponent's win
        action_index = action[0] * 3 + action[1] 
        return torch.tensor([[action_index]], device=device, dtype=torch.long)
    

    randam_value = random.random()

    if randam_value < epsilon:
        available_actions = [i for i in range(9) if state.flatten()[i] == 0]
        action = random.choice(available_actions)
        return torch.tensor([[action]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            q_values = policy_net(state)
        available_actions = [i for i in range(9) if state.flatten()[i] == 0]
        q_values_available = q_values[0, available_actions]
        action = available_actions[torch.argmax(q_values_available).item()]
        return torch.tensor([[action]], device=device, dtype=torch.long)