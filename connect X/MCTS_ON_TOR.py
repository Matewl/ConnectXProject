
import numpy as np

def make_turn(board, mark, turn):
    columns = 7
    rows = 6
    row = max([r for r in range(rows) if board[turn + (r * columns)] == 0])
    board[turn + (row * columns)] = mark
    
def bad_move(board, turn):
    columns = 7
    rows = 6
    if len([r for r in range(rows) if board[turn + (r * columns)] == 0]) == 0:
        return True
    return False

def is_win_tor(board, column, mark):
    columns = 7
    rows = 6
    current_board = np.array(board.copy())

    def move_board(board, r, c):
        new_board = np.zeros(board.shape)
        rows, cols = new_board.shape
        for i in range(rows):
            for j in range(cols):
                new_board[i][j] = board[(i + rows - r) % rows][(j + cols - c) % cols]
        return new_board
    
    def is_win_simple(board_, mark):
        """ Checks for a win. Taken from the Kaggle environment. """
        if board_[0][0] == mark and board_[0][1] == mark and board_[0][2] == mark and board_[0][3] == mark:
            return True
        if board_[0][0] == mark and board_[1][0] == mark and board_[2][0] == mark and board_[3][0] == mark:
            return True
        if board_[0][0] == mark and board_[1][1] == mark and board_[2][2] == mark and board_[3][3] == mark:
            return True
        if board_[3][0] == mark and board_[2][1] == mark and board_[1][2] == mark and board_[0][3] == mark:
            return True
        return False

    for k in range(rows):
        for m in range(columns):
            new_board = move_board(current_board.reshape(rows, columns), k, m)
            if is_win_simple(new_board, mark):
                return True
            
    return False


def is_tie(board):
    return not(any(mark == 0 for mark in board))

def get_reward(board, column, mark):
    if is_tie(board):
        return 0.5
    if is_win_tor(board, column, mark):
        return 1

    return 0 # игра еще не закончена 

def find_action_taken_by_opponent(new_board, old_board, config):
    """ Given a new board state and a previous one, finds which move was taken. Used for recycling tree between moves. """
    for i, piece in enumerate(new_board):
        if piece != old_board[i]:
            return i % config.columns
    return -1  # shouldn't get here

class MCTS_Node:
    def __init__(self, board, mark, terminal, game_result = None, parent = None, parent_action = None) -> None:
        self.board = board
        self.mark = mark
        self.terminal = terminal
        self.game_result = game_result
        self.parent = parent
        self.parent_action = parent_action
        self.children: list[MCTS_Node] = []
        self.number_visits = 0
        self.score = 0
        self.untried_actions = self.available_moves()

    def available_moves(self, board = None):
        if board is None:
            board = self.board

        return [move for move in range(7) if board[move] == 0]
    # def q(self):
    #     wins = self._results[1]
    #     loses = self._results[-1]
    #     return wins - loses
    def n(self):
        return self.number_visits

    def expand(self):
        action = self.untried_actions.pop()
        new_board = self.board.copy()

        make_turn(new_board, self.mark, action)

        score = get_reward(new_board, action, self.mark)
        terminal = True

        if score == 0:
            terminal = False
        child_node = MCTS_Node(
            new_board, mark=3 - self.mark, terminal=terminal, game_result=score, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node 
    
    def backpropagate(self, result):
        self.number_visits += 1.
        self.score += result
        if result == 0:
            self.score -= 5
        if self.parent:
            self.parent.backpropagate(1 - result)

    def rollout(self):
        if self.terminal:
            return self.game_result
        mark = self.mark
        new_board = self.board.copy()
        action = self.rollout_policy(self.available_moves())
        make_turn(new_board,  mark, action)
        score = get_reward(new_board, action, mark)

        while  score == 0:
            mark = 3 - mark
            action = self.rollout_policy(self.available_moves(new_board))

            make_turn(new_board, mark, action)
            score = get_reward(new_board, action, mark)
        if mark == self.mark:
            return score
        return 1 - score
    
    def rollout_policy(self, available_moves): # may be better?
        return available_moves[np.random.randint(len(available_moves))] #may be here

    def simulate(self):
        if self.terminal:
            return self.game_result
        return 1 - self.rollout()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=0.1):    
        choices_weights = [(c.score / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def best_action(self):
        simulation_no = 100
            
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)
    
    def expand_and_simulate_child(self):
        self.expand()
        simulation_score = self.children[-1].simulate()
        self.children[-1].backpropagate(simulation_score)

    def tree_single_run(self):
        if self.terminal:
            self.backpropagate(self.game_result)
            return
        if not self.is_fully_expanded():
            self.expand_and_simulate_child()
            return
        self.best_child().tree_single_run()
            
    def choose_child_via_action(self, action):
        for child in self.children:
            if child.parent_action == action:
                return child
        return None

def MCTS_on_tor_agent(observation, configuration):
    """
    Connect X agent based on MCTS.
    """
    import random
    import math
    import time
    global current_state  # so tree can be recycled
    board = observation.board
    mark = observation.mark
    init_time = time.time()
    EMPTY = 0
    T_max = 35  # time per move, left some overhead
    Cp_default = 13

    
    for turn in [move for move in range(7) if board[move] == 0]:
        new_board = board.copy()
        make_turn(new_board, 3 - mark, turn)
        if is_win_tor(new_board, turn, 3 - mark):
            return turn
    # If current_state already exists, recycle it based on action taken by opponent
    try:  
        current_state = current_state.choose_child_via_action(
            find_action_taken_by_opponent(board, current_state.board, configuration))
        current_state.parent = None  # make current_state the root node, dereference parents and siblings
        
    except:  # new game or other error in recycling attempt due to Kaggle mechanism
        current_state = MCTS_Node(board=board,mark= mark, terminal=False)
   
    # Run MCTS iterations until time limit is reached.
    while time.time() - init_time <= T_max:
        current_state.tree_single_run()
        
    current_state = current_state.best_child()
    return current_state.parent_action