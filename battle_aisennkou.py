import torch
from games.tictactoe import initialize_board, print_board, player_move, check_winner
from models.dqn import DQN
from trains.train import to_tensor

def prevent_win(board):
    """
    敵が勝利する手を防ぐ行動を特定する関数。
    """
    for i in range(3):
        row = board[i]
        if row.count('O') == 2 and row.count(' ') == 1:
            return i, row.index(' ')  # 空きセルの位置を返す
        col = [board[j][i] for j in range(3)]
        if col.count('O') == 2 and col.count(' ') == 1:
            return col.index(' '), i  # 空きセルの位置を返す

    # 斜めの確認
    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('O') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ')  # 空きセルの位置を返す

    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('O') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')  # 空きセルの位置を返す

    return None

def find_winning_move(board):
    """
    自分が勝利する手を特定する関数。
    """
    for i in range(3):
        row = board[i]
        if row.count('X') == 2 and row.count(' ') == 1:
            return i, row.index(' ')  # 勝利への最後の空きセルの位置を返す
        col = [board[j][i] for j in range(3)]
        if col.count('X') == 2 and col.count(' ') == 1:
            return col.index(' '), i  # 勝利への最後の空きセルの位置を返す

    # 斜めの確認
    diag1 = [board[i][i] for i in range(3)]
    if diag1.count('X') == 2 and diag1.count(' ') == 1:
        return diag1.index(' '), diag1.index(' ')  # 勝利への最後の空きセルの位置を返す

    diag2 = [board[i][2-i] for i in range(3)]
    if diag2.count('X') == 2 and diag2.count(' ') == 1:
        return diag2.index(' '), 2-diag2.index(' ')  # 勝利への最後の空きセルの位置を返す

    return None

def ai_move(board, model, device='cpu'):
    """
    AIが次の行動を決定する関数。
    """

    # 自分が勝利する手を最優先で選択
    winning_action = find_winning_move(board)
    if winning_action is not None:
        return winning_action

    # 敵が勝利する手を防ぐ行動を最優先で選択
    prevent_action = prevent_win(board)
    if prevent_action is not None:
        return prevent_action
    
    # 盤面をテンソルに変換し、適切なデバイスに移動
    state = to_tensor(board).to(device)
    # モデルを評価モードに設定
    model.eval()

    with torch.no_grad():
        q_values = model(state)
    
    # 盤面から利用可能なアクションを取得
    available_actions = [i for i, x in enumerate([cell for row in board for cell in row]) if x == ' ']

    # 利用可能なアクションのQ値のみを抽出
    q_values_available = q_values[0, available_actions]

    # 最もQ値が高いアクションを選択
    action = available_actions[torch.argmax(q_values_available).item()]

    # 選択したアクションを盤面の行と列に変換
    row, col = divmod(action, 3)
    print(f"row, col = {row}, {col}")
    return row, col

if __name__ == '__main__':
    # デバイスの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AIモデルの初期化と重みの読み込み
    ai_model = DQN().to(device)
    ai_model.load_state_dict(torch.load('trains/policy_net_1_final_weights_rlhf.pt', map_location=device))

    board = initialize_board()
    current_player = 1

    while True:
        print_board(board)
        if current_player == 1:
            print("Tatsuma AIの順番です")
            # AIの行動を取得
            row, col = ai_move(board, ai_model, device)
            # AIの行動を盤面に適用
            board[row][col] = 'X'
        else:
            # 人間のプレーヤーの入力を受け付け
            player_move(board, current_player)

        result = check_winner(board)
        if result != -1:
            print_board(board)
            if result == 0:
                print("引き分けです！")
            else:
                print(f"プレーヤー{result} [{ 'X' if result == 1 else 'O' }] の勝ちです！")
            break

        current_player = 2 if current_player == 1 else 1
