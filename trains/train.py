import torch
import numpy as np
from collections import namedtuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from games.tictactoe import initialize_board, print_board, player_move, check_winner
from trains.train_utils import select_action, optimize_model, ReplayMemory
from models.dqn import DQN

# その他の必要なインポートと変数の初期化

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
MEMORY_SIZE = 10000 

# 経験再生バッファで使うためのトランジションを定義
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def to_tensor(board):
    """盤面の状態をテンソルに変換する関数。' ' -> 0, 'X' -> 1, 'O' -> -1"""
    state = [0 if cell == ' ' else 1 if cell == 'X' else -1 for row in board for cell in row]
    return torch.tensor(state, dtype=torch.float).unsqueeze(0)  # バッチ次元を追加

def choose_action(state, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY, device):
    """エージェントが行動を選択する関数。ランダムまたはポリシーに基づく行動選択を行う。"""
    return select_action(state, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY, device)

def play_step(board, player, action):
    """選択した行動を盤面に適用し、次の状態、報酬、ゲームの終了フラグを返す関数。
    
    Args:
        board: 現在の盤面の状態。
        player: 現在のプレイヤー（1か2）。
        action: 行動（選択したセルの位置）。

    Returns:
        next_state: 行動後の盤面の状態。
        reward: その行動に対する報酬。
        done: ゲームが終了したかどうかのフラグ。
    """
    # 行動（セルの位置）から行と列を計算
    row, col = divmod(action.item(), 3)
    # 盤面にプレイヤーのシンボルを置く
    symbols = ['X', 'O']
    symbol = symbols[player - 1]
    board[row][col] = symbol

    # 盤面をNumPy配列に変換
    next_state = np.array(board)

    # 勝敗の判定
    winner = check_winner(next_state)

    # ゲームの終了フラグを設定（勝者がいるか引き分けの場合にゲーム終了）
    done = winner != -1
    # 報酬を計算
    reward = 0
    if winner == player:
        reward = 1 # 買った場合の報酬
    elif winner == 0:
        reward = 0.5  # 引き分けに対する報酬
    elif winner == 3 - player:
        reward = -1 # 負けた場合の報酬

    return next_state, reward, done

def train_dqn(num_episodes=10000):
    """DQNを用いてエージェントを訓練するメイン関数。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # モデルの初期化
    policy_net_1 = DQN().to(device)
    target_net_1 = DQN().to(device)
    target_net_1.load_state_dict(policy_net_1.state_dict())
    target_net_1.eval()
    
    policy_net_2 = DQN().to(device)
    target_net_2 = DQN().to(device)
    target_net_2.load_state_dict(policy_net_2.state_dict())
    target_net_2.eval()

    optimizer_1 = torch.optim.Adam(policy_net_1.parameters(), lr=0.001)
    optimizer_2 = torch.optim.Adam(policy_net_2.parameters(), lr=0.001)

    # 経験再生バッファの初期化
    memory = ReplayMemory(MEMORY_SIZE)

    BATCH_SIZE = 128
    GAMMA = 0.99
    TARGET_UPDATE = 10
    steps_done = 0

    for episode in range(num_episodes):
        # 盤面の初期化
        board = initialize_board()
        # 現在のプレーヤーの初期化
        current_player = 1
        # 初期化された盤面を初期化
        # tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        state = to_tensor(board)
        # 実行終了フラグの初期化
        done = False

        while not done:
            # エージェントに行動を選択させる
            # 現在のプレイヤーに応じて行動を選択
            if current_player == 1:
                state = state.to(device)
                action = choose_action(state, policy_net_1, steps_done, EPS_START, EPS_END, EPS_DECAY, device)
                optimizer = optimizer_1
                policy_net = policy_net_1
                target_net = target_net_1
            else:
                state = state.to(device)
                action = choose_action(state, policy_net_2, steps_done, EPS_START, EPS_END, EPS_DECAY, device)
                optimizer = optimizer_2
                policy_net = policy_net_2
                target_net = target_net_2

            # 学習の進行具合を示すフラグを更新
            steps_done += 1

            # stepの進行
            next_board, reward, done = play_step(board, current_player, action)
            # 次の状態をテンソルに変換
            next_state = to_tensor(next_board)
            
            # 経験をメモリに保存
            memory.push(state.to(device), action.to(device), next_state.to(device), torch.tensor([reward], dtype=torch.float, device=device))

            # 次の状態に更新
            state = next_state
            board = next_board.tolist()

            # モデルの最適化
            optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device)

            # ターンを交代
            current_player = 2 if current_player == 1 else 1

        # ターゲットネットワークの更新
        if episode % TARGET_UPDATE == 0:
            target_net_1.load_state_dict(policy_net_1.state_dict())
            target_net_2.load_state_dict(policy_net_2.state_dict())

        print(f"エピソード {episode} 完了")

    print("訓練完了")
    # エージェント1とエージェント2のネットワークの重みをファイルに保存
    torch.save(policy_net_1.state_dict(), 'policy_net_1_final_weights.pt')
    torch.save(policy_net_2.state_dict(), 'policy_net_2_final_weights.pt')

    torch.save(target_net_1.state_dict(), 'target_net_1_final_weights.pt')
    torch.save(target_net_2.state_dict(), 'target_net_2_final_weights.pt')

if __name__ == "__main__":
    train_dqn()
