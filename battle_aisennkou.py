import torch
from games.tictactoe import initialize_board, print_board, player_move, check_winner
from models.dqn import DQN
from trains.train import to_tensor

def ai_move(board, model, device='cpu'):
    """
    AIが次の行動を決定する関数。
    """
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
    ai_model.load_state_dict(torch.load('trains/policy_net_2_final_weights.pt', map_location=device))

    board = initialize_board()
    current_player = 1

    while True:
        print_board(board)
        if current_player == 1:
            print("AI君の順番です")
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
