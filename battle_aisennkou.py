import torch
from games.tictactoe import initialize_board, print_board, player_move, check_winner
from models.dqn import DQN
from trains.train import to_tensor

def prevent_win(board):
    """
    Identifies actions to prevent the opponent from winning.
    """
    for i in range(3):
        row = board[i]
        # Check if the opponent is one move away from winning in a row
        if row.count('O') == 2 and row.count(' ') == 1:
            return i, row.index(' ')  # Return the position of the empty cell
        col = [board[j][i] for j in range(3)]
        # Check if the opponent is one move away from winning in a column
        if col.count('O') == 2 and col.count(' ') == 1:
            return col.index(' '), i  # Return the position of the empty cell

    # Check diagonals
    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('O') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ')  # Return the position of the empty cell

    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('O') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')  # Return the position of the empty cell

    return None

def find_winning_move(board):
    """
    Identifies moves that could lead to a win.
    """
    for i in range(3):
        row = board[i]
        # Check if a winning move is available in a row
        if row.count('X') == 2 and row.count(' ') == 1:
            return i, row.index(' ')  # Return the position of the last empty cell for a win
        col = [board[j][i] for j in range(3)]
        # Check if a winning move is available in a column
        if col.count('X') == 2 and col.count(' ') == 1:
            return col.index(' '), i  # Return the position of the last empty cell for a win

    # Check diagonals
    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('X') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ')  # Return the position of the last empty cell for a win

    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('X') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')  # Return the position of the last empty cell for a win

    return None

def ai_move(board, model, device='cpu'):
    """
    Function that determines the next move for the AI.
    """

    # First priority is to select a move that results in a win
    winning_action = find_winning_move(board)
    if winning_action is not None:
        return winning_action # If such a move exists, return it

    # Next priority is to select a move that prevents the opponent from winning
    prevent_action = prevent_win(board)
    if prevent_action is not None:
        return prevent_action # If such a move exists, return it
    
    # Convert the board to a tensor and move it to the specified device (e.g., CPU)
    state = to_tensor(board).to(device)
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad(): # Do not compute gradients to speed up the computation
        q_values = model(state) # Get the Q-values from the model for the current state
    
    # Retrieve available actions from the board state
    available_actions = [i for i, x in enumerate([cell for row in board for cell in row]) if x == ' ']

    # Extract Q-values for available actions only
    q_values_available = q_values[0, available_actions]

    # Select the action with the highest Q-value
    action = available_actions[torch.argmax(q_values_available).item()]

    # Convert the selected action index back into board row and column
    row, col = divmod(action, 3)
    print(f"row, col = {row}, {col}")
    return row, col

if __name__ == '__main__':
    # Specify the device for computation (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the AI model and load its weights
    ai_model = DQN().to(device)
    ai_model.load_state_dict(torch.load('trains/policy_net_1_final_weights_rlhf.pt', map_location=device))

    board = initialize_board() # Initialize the game board
    current_player = 1 # Start with player 1

    while True:
        print_board(board) # Display the current state of the board
        if current_player == 1:
            print("AI's turn")
            # Get the AI's move
            row, col = ai_move(board, ai_model, device)
            # Apply the AI's move to the board
            board[row][col] = 'X'
        else:
            # Accept input from the human player
            player_move(board, current_player)

        result = check_winner(board) # Check for a winner
        if result != -1: # If the game has ended (win or draw)
            print_board(board) # Display the final state of the board
            if result == 0:
                print("It's a draw!")
            else:
                print(f"Player {result} [{'X' if result == 1 else 'O'}] wins!")
            break # Exit the game loop

        current_player = 2 if current_player == 1 else 1 # Switch the current player
