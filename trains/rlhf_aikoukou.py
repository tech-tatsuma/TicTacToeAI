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

# 学習状態を保存/読み込むためのファイル名
STATE_SAVE_PATH = 'training_state.json'

def save_training_state(steps_done, episode):
    """学習の進行状態を外部ファイルに保存する関数"""
    state = {
        'steps_done': steps_done,
        'episode': episode,
    }
    with open(STATE_SAVE_PATH, 'w') as f:
        json.dump(state, f)

def load_training_state():
    """外部ファイルから学習の進行状態を読み込む関数"""
    if os.path.exists(STATE_SAVE_PATH):
        with open(STATE_SAVE_PATH, 'r') as f:
            state = json.load(f)
        return state['steps_done'], state['episode']
    else:
        return 0, 0  # ファイルが存在しない場合は初期値を返す

# 変数の初期化
MEMORY_SIZE = 10000 

# 経験再生バッファで使うためのトランジションを定義
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def to_tensor(board):
    """盤面の状態をテンソルに変換する関数。' ' -> 0, 'X' -> 1, 'O' -> -1"""
    state = [0 if cell == ' ' else 1 if cell == 'X' else -1 for row in board for cell in row]
    return torch.tensor(state, dtype=torch.float).unsqueeze(0)

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
        reward = 1 # 勝った場合の報酬
    elif winner == 0:
        reward = 0.5  # 引き分けに対する報酬
    elif winner == 3 - player:
        reward = -1 # 負けた場合の報酬

    return next_state, reward, done

def evaluate_board(board, player):
    """盤面を評価して、ゲームが終了したかどうかと報酬を返す"""
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
    """DQNを用いてエージェントを訓練するメイン関数。"""

    # デバイスの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの初期化
    policy_net = DQN().to(device)
    policy_net.load_state_dict(torch.load('policy_net_2_final_weights.pt', map_location=device))
    target_net = DQN().to(device)
    target_net.load_state_dict(torch.load('target_net_2_final_weights.pt', map_location=device))
    target_net.eval()

    # オプティマイザの初期化
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    # 経験再生バッファの初期化
    memory = ReplayMemory(MEMORY_SIZE)

    # 保存された学習状態の読み込み
    steps_done, start_episode = load_training_state()

    BATCH_SIZE = 128
    GAMMA = 0.99
    TARGET_UPDATE = 10
    steps_done = 0

    for episode in range(start_episode, num_episodes):
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

            print_board(board)

            # エージェントに行動を選択させる
            # 現在のプレイヤーに応じて行動を選択
            if current_player == 1:
                # 人間との入力を受け付ける
                player_move(board, current_player)
                next_board = np.array(board)
                done, reward = evaluate_board(next_board, current_player)
            else:
                print('AIの順番です')
                # symbolの設定
                # エージェントが-1の場合はsymbol = -1
                symbol = -1
                # 盤面をstateに変換し，デバイスに転送
                state = to_tensor(board).to(device)
                # 行動の選択
                action = select_action_rlhf(state, policy_net, device, symbol)
                # 行動を適用し，次の状態，報酬，ゲームの終了フラグを取得
                next_board, reward, done = play_step(board, current_player, action)
                # 次の盤の状態をテンソルに変換
                next_state = to_tensor(next_board)
                # メモリに経験を追加
                memory.push(state, action, next_state, torch.tensor([reward], device=device))
                # 状態と盤面を更新
                state = next_state
                board = next_board
                # パラメータを更新
                optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device)
                torch.save(policy_net.state_dict(), 'policy_net_2_final_weights_rlhf.pt')

            # 学習の進行具合を示すフラグを更新
            steps_done += 1

            # ターンを交代
            current_player = 2 if current_player == 1 else 1

        # ターゲットネットワークの更新
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), 'target_net_2_final_weights_rlhf.pt')

        print_board(board)

        print(f"エピソード {episode} 完了")

        # エピソードごとに学習状態を保存
        save_training_state(steps_done, episode + 1)

    print("訓練完了")

if __name__ == "__main__":
    rlhf()