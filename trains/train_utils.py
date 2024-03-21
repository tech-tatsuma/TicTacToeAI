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
MEMORY_SIZE = 10000  # 経験再生バッファのサイズ．過去の経験をどれだけ保持するか．
BATCH_SIZE = 128  # バッチサイズ．一度に学習するデータの数．
LR = 0.001  # 学習率

# 経験再生バッファで使うためのトランジションを定義
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

def select_action(state, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY):
    # ランダムな値を取得
    sample = random.random()

    # ε-greedy法を用いて行動の選択閾値を計算
    # 初期段階ではランダムな行動を多く取り，時間が経つに連れて最適な行動をとるように変化する
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

    # ランダムな値が閾値よりも大きい場合は，ネットワークの予測に基づく行動を選択
    if sample > eps_threshold:
        with torch.no_grad():
            # ネットワークの出力から最大値を持つ行動を選択
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # ランダムな値が閾値以下の場合は，ランダムな行動を選択
        # ただし，選択可能な（まだ選ばれていない）セルのみから選ぶ
        available_actions = [i for i in range(9) if state.flatten()[i] == 0]
        if available_actions:
            # 空のセルがある場合は，その中からランダムに選択
            return torch.tensor([[random.choice(available_actions)]], dtype=torch.long)
        else:
            # 万が一全てのセルが埋まっている場合は，全範囲からランダムに選択（通常は発生しない）
            return torch.tensor([[random.randrange(9)]], dtype=torch.long)