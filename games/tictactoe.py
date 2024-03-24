def initialize_board():
    """
    Initializes the game board.
    """
    # Creates a 3x3 board filled with spaces, representing an empty board
    return [[' ' for _ in range(3)] for _ in range(3)]

def print_board(board):
    """
    Prints the current state of the board.
    """
    for row in board:
        # Joins the elements of the row with a vertical bar and prints it
        print("|".join(row))
        # Prints a horizontal line to separate rows
        print("-" * 5)

def player_move(board, player):
    """
    Allows a player to make a move by specifying a location.
    
    Args:
        board: The current state of the board.
        player: The current player (1 or 2).
    """
    # Assigns 'X' for player 1 and 'O' for player 2
    symbol = 'X' if player == 1 else 'O'
    while True:
        try:
            # Asks the player for a row and column, separated by a comma
            row, col = map(int, input(f"Player {player} [{symbol}]'s turn. Enter row and column separated by a comma (e.g., 1,2): ").split(','))
            # If the selected cell is empty, places the player's symbol there
            if board[row][col] == ' ':
                board[row][col] = symbol
                break
            else:
                # If the cell is already taken, prompts for a different location
                print("That location is already taken. Please choose another.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter the correct format.")

def check_winner(board):
    """
    Determines the winner of the game.
    
    Returns:
        1 if 'X' wins, 2 if 'O' wins, 0 for a draw, and -1 to continue the game.
    """
    # Lists all possible lines (rows, columns, diagonals) to check for a winner
    lines = [
        # Horizontal rows
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # Vertical columns
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # Diagonals
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]],
    ]
    
    for line in lines:
        # Checks if all elements in a line are the same and not empty
        if line[0] == line[1] == line[2] and line[0] != ' ':
            # Returns 1 for 'X' winning, 2 for 'O' winning
            return 1 if line[0] == 'X' else 2
    
    # Checks for a draw (if all cells are filled and no winner)
    if all(board[row][col] != ' ' for row in range(3) for col in range(3)):
        return 0  # Draw
    return -1  # Game continues