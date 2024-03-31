def print_board(board):
    print("\n|-----+-----+-----+-----+-----+-----+-----|")
    for i in range(6):
        print('|  ', end = '')
        for j in range(7):
            print(board[i * 7 + j], end='  |  ')
        print("\n|-----+-----+-----+-----+-----+-----+-----|")

def renderer(state, env):
    def print_board(board):
        for i in range(6):
            for j in range(7):
                print(board[i * 7 + j], end='   ')
            print('\n')
    print_board(state[0].observation.board)

def draw_board(board):
    s = ''
    s += "\n|-----+-----+-----+-----+-----+-----+-----|\n"
    for i in range(6):
        s += '|  '
        for j in range(7):
            s += str(board[i * 7 + j]) + '  |  '
        s += "\n|-----+-----+-----+-----+-----+-----+-----|\n"
    return '```' + s + '```'