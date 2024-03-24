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
    Identifies moves that could lead to a win.
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
    Function that determines the next move for the AI.
    """

    winning_action = find_winning_move(board)
    if winning_action is not None:
        return winning_action

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ai_model = DQN().to(device)
    ai_model.load_state_dict(torch.load('trains/policy_net_1_final_weights.pt', map_location=device))

    board = initialize_board()
    current_player = 1

    while True:
        print_board(board)
        if current_player == 2:
            print("AI's turn")

            row, col = ai_move(board, ai_model, device)
  
            board[row][col] = 'O'
        else:
            player_move(board, current_player)

        result = check_winner(board)
        if result != -1:
            print_board(board)
            if result == 0:
                print("It's a draw!")
            else:
                print(f"Player {result} [{ 'X' if result == 1 else 'O' }] wins!")
            break

        current_player = 2 if current_player == 1 else 1
