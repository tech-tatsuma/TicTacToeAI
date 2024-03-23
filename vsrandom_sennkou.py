import torch
import random
from collections import defaultdict

from games.tictactoe import initialize_board, check_winner
from models.dqn import DQN
from trains.train import to_tensor

def random_move(board, player):
    """Make a random move for a player on the board.

    Args:
        board: The current state of the board as a list of lists.
        player: The current player (1 or 2).
    """
    # Assign 'X' to player 1 and 'O' to player 2.
    symbol = 'X' if player == 1 else 'O'

    # Find all empty positions on the board.
    empty_positions = [(r, c) for r in range(len(board)) for c in range(len(board[r])) if board[r][c] == ' ']

    # If there are empty positions, choose one at random and place the player's symbol there.
    if empty_positions:
        row, col = random.choice(empty_positions)
        board[row][col] = symbol
    else:
        # If there are no empty positions, print a message indicating the board is full.
        print("The board is full.")

def prevent_win(board):
    """
    Identifies a move to prevent the opponent from winning.
    
    This function checks the board for any potential win situations for the opponent ('O') 
    and returns a move that blocks the opponent from winning if such a situation is found.
    
    Returns:
        A tuple (row, column) indicating the position to block the opponent's win, 
        or None if no such position exists.
    """
    # Check each row for a potential win situation for 'O'.
    for i in range(3):
        row = board[i]
        # If there are two 'O's and one empty space in a row, return the position to block.
        if row.count('O') == 2 and row.count(' ') == 1:
            return i, row.index(' ')  # Return the position of the empty cell.

        # Check each column for a potential win situation for 'O'.
        col = [board[j][i] for j in range(3)]
        # If there are two 'O's and one empty space in a column, return the position to block.
        if col.count('O') == 2 and col.count(' ') == 1:
            return col.index(' '), i  # Return the position of the empty cell.

    # Check the first diagonal for a potential win situation for 'O'.
    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('O') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ')  # Return the position of the empty cell.

    # Check the second diagonal for a potential win situation for 'O'.
    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('O') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')  # Return the position of the empty cell.

    # If no potential win situation for 'O' is found, return None.
    return None

def find_winning_move(board):
    """
    Identify moves to prevent the opponent from winning.
    
    Args:
        board: The current state of the board as a list of lists.
    
    Returns:
        Tuple containing the row and column of the blocking move, or None if no blocking move is necessary.
    """
    # Check each row for a potential win for the opponent.
    for i in range(3):
        row = board[i]
        # If there are two 'O's and one empty space, return the position to block.
        if row.count('X') == 2 and row.count(' ') == 1:
            return i, row.index(' ')  # Return the position of the empty cell to block.
        
        # Check each column for a potential win for the opponent.
        col = [board[j][i] for j in range(3)]
        if col.count('X') == 2 and col.count(' ') == 1:
            return col.index(' '), i  # Return the position of the empty cell to block.

    # Check the diagonal from top-left to bottom-right for a potential win.
    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('X') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ')  # Return the position of the empty cell to block.

    # Check the diagonal from top-right to bottom-left for a potential win.
    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('X') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')  # Return the position of the empty cell to block.

    # If no blocking move is found, return None.
    return None

def ai_move(board, model, device='cpu'):
    """
    Function to determine the next move by AI.

    Args:
        board: The current state of the board as a list of lists.
        model: The neural network model to evaluate the best move.
        device: The device (cpu or cuda) on which to run the model.
    
    Returns:
        A tuple (row, col) indicating the AI's chosen move.
    """

    # First, check if there's a move that leads to victory and choose it.
    winning_action = find_winning_move(board)
    if winning_action is not None:
        return winning_action

    # Next, check if the opponent is about to win and block them.
    prevent_action = prevent_win(board)
    if prevent_action is not None:
        return prevent_action
    
    # Convert the board state to a tensor, moving it to the specified device.
    state = to_tensor(board).to(device)
    # Set the model to evaluation mode.
    model.eval()

    with torch.no_grad():
        # Predict the Q-values for each action using the model.
        q_values = model(state)
    
    # Extract available actions based on the current board state.
    available_actions = [i for i, x in enumerate([cell for row in board for cell in row]) if x == ' ']

    # Select the Q-values for the available actions.
    q_values_available = q_values[0, available_actions]

    # Choose the action with the highest Q-value.
    action = available_actions[torch.argmax(q_values_available).item()]

    # Convert the chosen action into row and column indices.
    row, col = divmod(action, 3)
    print(f"row, col = {row}, {col}")
    return row, col

def play_game(ai_model, device):
    # Initialize the board for a new game.
    board = initialize_board()
    # Start with player 1.
    current_player = 1

    # Loop until the game ends.
    while True:
        # Player 1's turn.
        if current_player == 1:
            # Use the AI to determine the move.
            row, col = ai_move(board, ai_model, device)
            # Mark the board with player 1's symbol ('X').
            board[row][col] = 'X'
        else:
            # Player 2's turn, using a random move strategy.
            random_move(board, current_player)

        # Check if there's a winner after the recent move.
        result = check_winner(board)
        # If there's a winner or the game is a draw, end the game.
        if result != -1:
            return result

        # Switch to the other player for the next turn.
        current_player = 2 if current_player == 1 else 1

if __name__ == '__main__':
    # Set up the device to use for PyTorch (use GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the AI model and move it to the appropriate device.
    ai_model = DQN().to(device)
    # Load the trained model weights from a file.
    ai_model.load_state_dict(torch.load('trains/policy_net_1_final_weights_rlhf.pt', map_location=device))

    # Dictionary to keep track of game outcomes.
    results = defaultdict(int)

    # Play 1000 games between the AI and its opponent.
    for _ in range(1000):
        # Play a game and get the winner.
        winner = play_game(ai_model, device)
        # Increment the count for the winning player or for a draw.
        results[winner] += 1

    # Print the results: number of games won by the AI, drawn, and lost by the AI.
    print(f"Match Results: AI wins: {results[1]}, Draws: {results[0]}, AI losses: {results[2]}")