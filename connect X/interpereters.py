EMPTY = 0

from MCTS import MCTS_agent, is_win
from MCTS_ON_TOR import MCTS_on_tor_agent, make_turn, bad_move, is_win_tor, is_tie

def interpreter(state, env):
    active = state[0] if state[0].status == "ACTIVE" else state[1]

    # Specification can fully handle the reset.
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # Keep the board in sync between both agents.
    board = active.observation.board
    inactive.observation.board = board
    action = active.action
    mark = active.observation.mark

    # Illegal move by the active agent.
    if bad_move(board, action):
        active.status = f"Invalid move: {action}"
        inactive.status = "DONE"
        return state

    # Mark the position.
    make_turn(board, mark, action)

    # Check for a win.
    if is_win(board, action, mark):
        active.reward = 1
        active.status = "DONE"
        inactive.reward = 0
        inactive.status = "DONE"
    
        return state

    # Check for a tie.
    if is_tie(board):
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    # Swap active and inactive agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state

def interpreter_tor(state, env):
    active = state[0] if state[0].status == "ACTIVE" else state[1]

    # Specification can fully handle the reset.
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # Keep the board in sync between both agents.
    board = active.observation.board
    inactive.observation.board = board
    action = active.action
    mark = active.observation.mark

    # Illegal move by the active agent.
    if bad_move(board, action):
        active.status = f"Invalid move: {action}"
        inactive.status = "DONE"
        return state

    # Mark the position.
    make_turn(board, mark, action)

    # Check for a win.
    if is_win_tor(board, action, mark):
        active.reward = 1
        active.status = "DONE"
        inactive.reward = 0
        inactive.status = "DONE"
    
        return state

    # Check for a tie.
    if is_tie(board):
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    # Swap active and inactive agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state