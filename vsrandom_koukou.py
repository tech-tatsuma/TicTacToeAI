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
    symbol = 'X' if player == 1 else 'O'
    empty_positions = [(r, c) for r in range(len(board)) for c in range(len(board[r])) if board[r][c] == ' ']
    if empty_positions:
        row, col = random.choice(empty_positions)
        board[row][col] = symbol
    else:
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
    for i in range(3):
        row = board[i]
        if row.count('X') == 2 and row.count(' ') == 1:
            return i, row.index(' ') 
        col = [board[j][i] for j in range(3)]
        if col.count('X') == 2 and col.count(' ') == 1:
            return col.index(' '), i 

    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('X') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ') 

    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('X') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')

    return None

def find_winning_move(board):
    """
    Identify moves to prevent the opponent from winning.
    
    Args:
        board: The current state of the board as a list of lists.
    
    Returns:
        Tuple containing the row and column of the blocking move, or None if no blocking move is necessary.
    """
    for i in range(3):
        row = board[i]
        if row.count('O') == 2 and row.count(' ') == 1:
            return i, row.index(' ')
        col = [board[j][i] for j in range(3)]
        if col.count('O') == 2 and col.count(' ') == 1:
            return col.index(' '), i 

    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('O') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ') 

    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('O') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')

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
    
    state = to_tensor(board).to(device)
    model.eval()

    with torch.no_grad():
        q_values = model(state)
    
    available_actions = [i for i, x in enumerate([cell for row in board for cell in row]) if x == ' ']

    q_values_available = q_values[0, available_actions]

    action = available_actions[torch.argmax(q_values_available).item()]

    row, col = divmod(action, 3)
    print(f"row, col = {row}, {col}")
    return row, col

def play_game(ai_model, device):
    board = initialize_board()
    current_player = 1

    while True:
        if current_player == 1:
            random_move(board, current_player)
        else:
            row, col = ai_move(board, ai_model, device)
            board[row][col] = 'O'

        result = check_winner(board)
        if result != -1:
            return result

        current_player = 2 if current_player == 1 else 1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = DQN().to(device)
    ai_model.load_state_dict(torch.load('trains/policy_net_2_final_weights_rlhf.pt', map_location=device))

    results = defaultdict(int) 

    for _ in range(1000):
        winner = play_game(ai_model, device)
        results[winner] += 1

    print(f"Match Results: AI wins: {results[2]}, Draws: {results[0]}, AI losses: {results[1]}")