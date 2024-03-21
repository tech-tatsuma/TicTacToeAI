import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
from models.dqn import DQN
import numpy as np

# 学習用パラメータの設定
GAMMA = 0.99  # 割引率．将来の報酬をどれだけ価値あるものとみなすかを決定．
EPS_START = 0.9  # 探索の開始確率．最初はランダムに行動を選択する確率が高くなる．
EPS_END = 0.05  # 探索の最小確率．時間が経つにつれてランダムに行動を選択する確率を減らす．
EPS_DECAY = 200  # 探索率の減衰率．この値が大きいほど探索期間が長くなる．
TARGET_UPDATE = 10  # ターゲットネットワークの更新間隔（エピソード数）
MEMORY_SIZE = 15000  # 経験再生バッファのサイズ．過去の経験をどれだけ保持するか．
BATCH_SIZE = 128  # バッチサイズ．一度に学習するデータの数．
LR = 0.001  # 学習率

# 経験再生バッファで使うためのトランジションを定義
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def prevent_win(board):
    """
    敵が勝利する手を防ぐ行動を特定する関数。この修正版ではboardが数値型のnumpy.ndarrayであると想定。
    """
    for i in range(3):
        row = board[i, :]
        if np.count_nonzero(row == 2) == 2 and np.count_nonzero(row == 0) == 1:
            return i, np.where(row == 0)[0][0]  # 空きセルの位置を返す
        col = board[:, i]
        if np.count_nonzero(col == 2) == 2 and np.count_nonzero(col == 0) == 1:
            return np.where(col == 0)[0][0], i  # 空きセルの位置を返す

    # 斜めの確認
    diag1 = board.diagonal()
    if np.count_nonzero(diag1 == 2) == 2 and np.count_nonzero(diag1 == 0) == 1:
        return np.where(diag1 == 0)[0][0], np.where(diag1 == 0)[0][0]  # 空きセルの位置を返す

    diag2 = np.fliplr(board).diagonal()
    if np.count_nonzero(diag2 == 2) == 2 and np.count_nonzero(diag2 == 0) == 1:
        return np.where(diag2 == 0)[0][0], 2 - np.where(diag2 == 0)[0][0]  # 空きセルの位置を返す

    return None

class ReplayMemory(object):
    """経験再生バッファ"""

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity) # 指定されたサイズのdequeを作成

    def push(self, *args):
        """transitionをメモリに保存"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ランダムにバッチをサンプリングして返す"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """メモリの長さ（保存されているトランジションの数）を返す"""
        return len(self.memory)

def optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, device):
    """学習を最適化するための関数．ネットワークの重みを更新"""
    # メモリに十分なトランジション（経験）が蓄積されていなければ，何もせずに関数を終了
    if len(memory) < BATCH_SIZE:
        return
    # 訓練モードに設定
    policy_net.train()
    
    # 経験再生メモリからランダムにトランジションのバッチをサンプリング
    transitions = memory.sample(BATCH_SIZE)

    # トランジションのバッチを古語のコンポーネント（状態，行動，次の状態，報酬）に分解
    batch = Transition(*zip(*transitions))

    # ゲームが終了していない（つまり次の状態が存在するトランジションを抽出）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    # 状態、行動、報酬のバッチをテンソルにまとめる
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    policy_net = policy_net.to(device)

    # 現在の行動に対するQ値を計算
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 次の状態における最大のQ値を計算（ゲーム終了時は0とする）
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # 期待されるQ値（ターゲット値）を計算
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber損失を用いて、予測されるQ値と期待されるQ値の差を計算
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 損失を用いてモデルの最適化を行う
    optimizer.zero_grad() # 勾配を0にリセット
    loss.backward() # バックプロパゲーションを用いて勾配を計算
    optimizer.step() # パラメータを更新

def select_action(state, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY, device):

    # stateを3x3の盤面形式に変換
    board = state.view(3, 3).cpu().numpy()

    # 敵の勝利を防ぐ手があるか確認
    action = prevent_win(board)
    if action is not None:
        # 敵の勝利を防ぐ行動を返す
        return torch.tensor([[action]], device=device, dtype=torch.long)
    
    # ランダムな値を取得
    sample = random.random()

    # ε-greedy法を用いて行動の選択閾値を計算
    # 初期段階ではランダムな行動を多く取り，時間が経つに連れて最適な行動をとるように変化する
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

    # ランダムな値が閾値よりも大きい場合は，ネットワークの予測に基づく行動を選択
    with torch.no_grad():
        q_values = policy_net(state)

    available_actions = [i for i in range(9) if state.flatten()[i] == 0]
    q_values_available = q_values[0, available_actions]

    if sample > eps_threshold:
        # 空のセルの中で一番勝つ確率が高いものを選択
        action = available_actions[torch.argmax(q_values_available).item()]
    else:
        # ランダムに行動を選択（空いているセルから）
        action = random.choice(available_actions)

    return torch.tensor([[action]], device=device, dtype=torch.long)