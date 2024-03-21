from games.tictactoe import initialize_board, print_board, player_move, check_winner

if __name__=='__main__':
    # 三目並べの初期化
    board = initialize_board()
    # カレントプレーヤーの初期化
    current_player = 1
    # 勝敗が決まるまでループ
    while True:
        # 盤面の状況を表示
        print_board(board)
        # プレーヤーの入力を受け付け
        player_move(board, current_player)
        # 勝敗の判定
        result = check_winner(board)
        # 勝敗が決まった場合
        if result != -1:
            print_board(board)
            # resultが0の時は引き分け
            if result == 0:
                print("引き分けです！")
            else:
                print(f"プレーヤー{result} [{ 'X' if result == 1 else 'O' }] の勝ちです！")
            break
        # ターンを交代
        current_player = 2 if current_player == 1 else 1