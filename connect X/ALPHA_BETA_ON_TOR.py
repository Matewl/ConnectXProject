def alphabeta_on_tor_agent(obs, config):
    import random
    import numpy as np
    


    # Get the number of pieces of the same mark in a window
    def move_board(board, r, c):
        new_board = np.zeros(board.shape)
        rows, cols = new_board.shape
        for i in range(rows):
            for j in range(cols):
                new_board[i][j] = board[(i + rows - r) % rows][(j + cols - c) % cols]
        return new_board

    # Get the number of pieces of the same mark in a window
    def pieces_in_window(window, piece):
        return window.count(piece) * (window.count(piece) + window.count(0) == config.inarow)
        # Calculates value of heuristic for grid
    
    def count_windows(board):
        grid = np.array(board).reshape((6, 7))
    
        windows = {piece: [0 for i in range(config.inarow+1)] for piece in [1, 2]}
        for r in range(6):
            for c in range(7):
                new_grid = move_board(grid, r, c)
                col = 0
                row = 0
                window = list(new_grid[row, col:col+config.inarow])
                windows[1][pieces_in_window(window, 1)]+=1
                windows[2][pieces_in_window(window, 2)]+=1

                window = list(new_grid[row:row+config.inarow, col])
                windows[1][pieces_in_window(window, 1)]+=1
                windows[2][pieces_in_window(window, 2)]+=1

                window = list(new_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                windows[1][pieces_in_window(window, 1)]+=1
                windows[2][pieces_in_window(window, 2)]+=1

                window = list(new_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                windows[1][pieces_in_window(window, 1)]+=1
                windows[2][pieces_in_window(window, 2)]+=1
        return windows

    def get_heuristic(board, mark):
        windows=count_windows(board)
        score =  windows[mark][1] + windows[mark][2]*3 + windows[mark][3]*9 + windows[mark][4]*81 - windows[mark%2+1][1] - windows[mark%2+1][2]*3 - windows[mark%2+1][3]*9 - windows[mark%2+1][4]*81
        return score
    
    def game_over(board):
        windows=count_windows(board)
        return windows[1][4] + windows[2][4] > 0
    
    def possible_turns(board):
        # print(board)
        # print([turn for turn in range(7) if board[turn] == 0])
        return [turn for turn in range(7) if board[turn] == 0]
    
    def make_turn(board, mark, turn):
        columns = 7
        rows = 6
        row = max([r for r in range(rows) if board[turn + (r * columns)] == 0])
        board[turn + (row * columns)] = mark
        return board

    # Alpha Beta pruning implementation

    def minimax(position, depth, alpha, beta, maximizing_player):
        board, mark = position[0], position[1]
        if depth == 0 or game_over(board):
            return get_heuristic(board, mark)
        if maximizing_player:
            max_eval = -np.inf
            for turn in possible_turns(board):
                new_board = board.copy()
                make_turn(new_board, mark, turn)
                child = (new_board, mark)
                eval = minimax(child, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for turn in possible_turns(board):
                new_board = board.copy()
                child = (make_turn(new_board, mark%2+1, turn), mark)
                eval = minimax(child, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval
        
    board = obs['board']
    mark = obs['mark']
    turns = possible_turns(board)

    n_steps = 4 if obs.board.count(0)>len(obs.board)*2/3 else 5 if obs.board.count(0)>len(obs.board)/3 else 6
    best_turn= None
    best_value = -np.inf
    for turn in turns:
        new_board = board.copy()
        make_turn(new_board, mark, turn)
        turn_value = minimax((new_board, mark), n_steps - 1, -np.inf, np.inf, False)
        if best_value < turn_value:
            best_value = turn_value
            best_turn = turn 
    return best_turn

