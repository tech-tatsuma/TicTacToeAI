def initialize_board():
    """盤面を初期化する関数。"""
    return [[' ' for _ in range(3)] for _ in range(3)]

def print_board(board):
    """盤面を表示する関数。"""
    for row in board:
        print("|".join(row))
        print("-" * 5)

def player_move(board, player):
    """プレーヤーが場所を指定する関数。
    
    Args:
        board: 現在の盤面。
        player: 現在のプレーヤー（1か2）。
    """
    symbol = 'X' if player == 1 else 'O'
    while True:
        try:
            row, col = map(int, input(f"プレーヤー{player} [{symbol}] の番です。行と列をカンマ区切りで入力してください（例：1,2）: ").split(','))
            if board[row][col] == ' ':
                board[row][col] = symbol
                break
            else:
                print("その場所は既に選ばれています。別の場所を選んでください。")
        except (ValueError, IndexError):
            print("無効な入力です。正しい形式で入力してください。")

def check_winner(board):
    """勝敗の判定を行う関数。
    
    Returns:
        'X'が勝った場合は1, 'O'が勝った場合は2, 引き分けは0, 続行は-1。
    """
    lines = [
        # 横の行
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # 縦の列
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # 斜め
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]],
    ]
    
    for line in lines:
        if line[0] == line[1] == line[2] and line[0] != ' ':
            return 1 if line[0] == 'X' else 2
    
    if all(board[row][col] != ' ' for row in range(3) for col in range(3)):
        return 0  # 引き分け
    return -1  # 続行