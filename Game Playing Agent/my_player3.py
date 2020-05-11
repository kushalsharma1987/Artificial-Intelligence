# 3225872363:Kushal Sharma #

import random
import time
from copy import deepcopy
import sys
import heapq
from json import JSONDecodeError

import numpy as np
import json

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0


class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        # self.previous_board = None # Store the previous board
        self.X_move = True  # X chess plays first
        self.died_pieces = []  # Intialize died pieces to be empty
        self.count_died_pieces = {}
        self.count_died_pieces[1] = 0
        self.count_died_pieces[2] = 0
        self.n_move = 0  # Trace the number of moves
        self.max_move = (n * n) - 1  # The max movement of a Go game
        self.komi = n / 2  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player

    def set_go_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_go_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def encode_go_board(self):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(self.board[i][j]) for i in range(self.size) for j in range(self.size)])

    def copy_go_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_my_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_my_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_my_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs_search(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_my_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_my_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs_search(i, j)
        for member in ally_members:
            neighbors = self.detect_my_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces_opp(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_my_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces_opp(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces_opp(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces_position(died_pieces)
        return died_pieces

    def remove_certain_pieces_position(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_go_board(board)

    def check_valid_move(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False

        # Copy the board for testing
        test_go = self.copy_go_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_go_board(test_board)
        if test_go.find_my_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces_opp(3 - piece_type)
        if not test_go.find_my_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_go_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True

    def update_go_board(self, new_board):
        '''
        Update the board with new_board
        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def get_final_score(self, piece_type):
        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        X_liberty = 0
        O_liberty = 0
        test_go = self.copy_go_board()
        board = test_go.board
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    board[i][j] = 1
                    test_go.update_go_board(board)
                    if test_go.find_my_liberty(i, j):
                        X_liberty += 1
                    board[i][j] = 2
                    test_go.update_go_board(board)
                    if test_go.find_my_liberty(i, j):
                        O_liberty += 1
                    board[i][j] = 0

        if piece_type == 1:
            final_score = cnt_1 + X_liberty + self.count_died_pieces[1] - cnt_2 - self.komi - O_liberty - \
                          self.count_died_pieces[2]
        elif piece_type == 2:
            final_score = cnt_2 + self.komi + O_liberty + self.count_died_pieces[2] - cnt_1 - X_liberty - \
                          self.count_died_pieces[1]

        return final_score


class AlphaBetaPlayer:

    def __init__(self):

        self.type = 'alphabeta'

    # Player 'X' is max
    def max_alpha_beta(self, go, orig_piece_type, piece_type, start_time, depth, moves_chg_depth, alpha, beta):

        curr_time = time.time()
        maxv = -np.inf
        px = None
        py = None

        d = 3
        if moves_chg_depth >= 14:
            d = 4

        if go.n_move >= go.max_move or round(curr_time - start_time, 7) > 9.00 or depth > d:
            # if go.n_move >= go.max_move or round(curr_time - start_time, 7) > 8.00:
            # if go.n_move >= go.max_move or depth > d:
            # if go.n_move >= go.max_move:
            final_score = go.get_final_score(orig_piece_type)
            return (final_score, 0, 0)

        m = "PASS"
        valid_skipped_moves = []
        for i in range(0, 5):
            for j in range(0, 5):
                if go.check_valid_move(i, j, piece_type):
                    # test_go = go.copy_go_board()
                    # test_board = test_go.board
                    # test_board[i][j] = 3 - piece_type
                    # if not test_go.find_my_liberty(i, j):
                    #    valid_skipped_moves.append((i, j))
                    #    del test_go
                    #    continue

                    new_go = go.copy_go_board()
                    new_board = new_go.board
                    new_board[i][j] = piece_type
                    new_go.update_go_board(new_board)
                    new_go.died_pieces = new_go.remove_died_pieces_opp(3 - piece_type)
                    new_go.count_died_pieces[3 - piece_type] += len(new_go.died_pieces)
                    new_go.n_move += 1
                    depth += 1
                    new_piece_type = 3 - piece_type
                    (m, min_i, min_j) = self.min_alpha_beta(new_go, orig_piece_type, new_piece_type, start_time, depth,
                                                            moves_chg_depth, alpha, beta)

                    if m > maxv:
                        maxv = m
                        px = i
                        py = j
                    del new_go
                    depth -= 1

                    # Next two ifs in Max and Min are the only difference between regular algorithm and minimax
                    if maxv >= beta:
                        return (maxv, px, py)

                    if maxv > alpha:
                        alpha = maxv
        if m == "PASS":
            return (go.get_final_score(orig_piece_type), px, py)
        return (maxv, px, py)

    # Player 'X' is min, in this case human
    def min_alpha_beta(self, go, orig_piece_type, piece_type, start_time, depth, moves_chg_depth, alpha, beta):

        curr_time = time.time()
        minv = np.inf
        qx = None
        qy = None

        d = 3
        if moves_chg_depth >= 14:
            d = 4

        if go.n_move >= go.max_move or round(curr_time - start_time, 7) > 9.00 or depth > d:
            # if go.n_move >= go.max_move or round(curr_time - start_time, 7) > 8.00:
            # if go.n_move >= go.max_move or depth > d:
            final_score = go.get_final_score(orig_piece_type)
            return (final_score, 0, 0)

        m = "PASS"
        valid_skipped_moves = []
        for i in range(0, 5):
            for j in range(0, 5):
                if go.check_valid_move(i, j, piece_type):
                    # test_go = go.copy_go_board()
                    # test_board = test_go.board
                    # test_board[i][j] = 3 - piece_type
                    # if not test_go.find_my_liberty(i, j):
                    #    valid_skipped_moves.append((i, j))
                    #    del test_go
                    #    continue

                    new_go = go.copy_go_board()
                    new_board = new_go.board
                    new_board[i][j] = piece_type
                    new_go.update_go_board(new_board)
                    new_go.died_pieces = new_go.remove_died_pieces_opp(3 - piece_type)
                    new_go.count_died_pieces[3 - piece_type] += len(new_go.died_pieces)
                    new_go.n_move += 1
                    depth += 1
                    new_piece_type = 3 - piece_type
                    (m, max_i, max_j) = self.max_alpha_beta(new_go, orig_piece_type, new_piece_type, start_time, depth,
                                                            moves_chg_depth, alpha, beta)

                    if m < minv:
                        minv = m
                        qx = i
                        qy = j
                    del new_go
                    depth -= 1

                    if minv <= alpha:
                        return (minv, qx, qy)
                    if minv < beta:
                        beta = minv
        if m == "PASS":
            return (go.get_final_score(orig_piece_type), qx, qy)

        return (minv, qx, qy)

    def get_input(self, go, piece_type, opp_move):
        """ make a move
        """
        start_time = time.time()
        board = go.board
        x, y = opp_move
        if go.n_move < 14:
            if board[2][2] == 0 and go.check_valid_move(2, 2, piece_type):
                return 2, 2
            if board[1][1] == 0 and go.check_valid_move(1, 1, piece_type):
                return 1, 1
            if board[1][3] == 0 and go.check_valid_move(1, 3, piece_type):
                return 1, 3
            if board[3][1] == 0 and go.check_valid_move(3, 1, piece_type):
                return 3, 1
            if board[3][3] == 0 and go.check_valid_move(3, 3, piece_type):
                return 3, 3
            if (x, y) != (None, None):
                score_heap = []
                if go.check_valid_move(x - 1, y, piece_type):
                    move = x - 1, y
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x - 1, y + 1, piece_type):
                    move = x - 1, y + 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x, y + 1, piece_type):
                    move = x, y + 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x + 1, y + 1, piece_type):
                    move = x + 1, y + 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x + 1, y, piece_type):
                    move = x + 1, y
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x + 1, y - 1, piece_type):
                    move = x + 1, y - 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x, y - 1, piece_type):
                    move = x, y - 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.check_valid_move(x - 1, y - 1, piece_type):
                    move = x - 1, y - 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))

                heapq._heapify_max(score_heap)
                if score_heap:
                    placement = heapq._heappop_max(score_heap)
                    possible_placements = []
                    max_score = placement[0]
                    possible_placements.append(placement[1])
                    while score_heap:
                        placement = heapq._heappop_max(score_heap)
                        if max_score != placement[0]:
                            break
                        possible_placements.append(placement[1])
                    row, col = random.choice(possible_placements)
                    return row, col

        test_go = go.copy_go_board()

        depth = 0
        moves_chg_depth = go.n_move
        orig_piece_type = piece_type

        (m, i, j) = self.max_alpha_beta(test_go, orig_piece_type, piece_type, start_time, depth, moves_chg_depth,
                                        -np.inf, np.inf)

        if (i, j) == (None, None):
            return "PASS"
        else:
            return i, j

    def alpha_beta_heuristic(self, go, move, piece_type, start_time):

        test_go = go.copy_go_board()
        test_board = test_go.board
        x, y = move
        test_board[x][y] = piece_type
        test_go.update_go_board(test_board)
        test_go.died_pieces = test_go.remove_died_pieces_opp(3 - piece_type)
        test_go.n_move += 1
        moves_chg_depth = test_go.n_move
        new_piece_type = 3 - piece_type
        orig_piece_type = piece_type
        depth = 1
        (m, i, j) = self.max_alpha_beta(test_go, orig_piece_type, new_piece_type, start_time, depth,
                                        moves_chg_depth, -np.inf, np.inf)
        return m

class QLearner:

    def __init__(self, alpha=.7, gamma=.9, initial_value=0.5, side=None):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value
        # self.state = ?
        self.type = 'qlearner'

    def load_qvalues_file(self, piece_type):
        if piece_type == 1:
            try:
                with open('qvalues_black.json', 'r') as infile:
                    return json.load(infile)
            except (EnvironmentError, JSONDecodeError):
                return {}
        elif piece_type == 2:
            try:
                with open('qvalues_white.json', 'r') as infile:
                    return json.load(infile)
            except (EnvironmentError, JSONDecodeError):
                return {}

    def Q(self, go, state):
        if state not in self.q_values:
            q_val = [[self.initial_value for i in range(go.size)] for j in range(go.size)]
            self.q_values[state] = q_val
        return self.q_values[state]

    def select_best_move(self, go, piece_type, opp_move):
        start_time = time.time()
        state = go.encode_go_board()
        q_values = self.Q(go, state)
        curr_max = -np.inf
        max_qvalue_nodes = []
        for i in range(go.size):
            for j in range(go.size):
                # if not state_present:
                # if not go.check_valid_move(i, j, piece_type):
                #     q_values[i][j] = -1.0
                if go.check_valid_move(i, j, piece_type):
                    if q_values[i][j] > curr_max:
                        curr_max = q_values[i][j]
                        max_qvalue_nodes.clear()
                        max_qvalue_nodes.append((i, j))
                    elif q_values[i][j] == curr_max:
                        max_qvalue_nodes.append((i, j))
                else:
                    q_values[i][j] = -1.0

        if not max_qvalue_nodes:
            curr_time = time.time()
            print('Evaluation time: {}s'.format(round(curr_time - start_time, 7)))
            print(sys._getframe().f_lineno)
            return "PASS"

        score_heap = []
        x, y = opp_move
        if max_qvalue_nodes:

            if go.n_move < 14:
                if (2, 2) in max_qvalue_nodes:
                    return 2, 2
                if (1, 1) in max_qvalue_nodes:
                    return 1, 1
                if (1, 3) in max_qvalue_nodes:
                    return 1, 3
                if (3, 1) in max_qvalue_nodes:
                    return 3, 1
                if (3, 3) in max_qvalue_nodes:
                    return 3, 3

                if (x, y) != (None, None):
                    value_heap = []
                    if (x-1, y) in max_qvalue_nodes:
                        move = x - 1, y
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x - 1, y + 1) in max_qvalue_nodes:
                        move = x - 1, y + 1
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x, y + 1) in max_qvalue_nodes:
                        move = x, y + 1
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x + 1, y + 1) in max_qvalue_nodes:
                        move = x + 1, y + 1
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x + 1, y) in max_qvalue_nodes:
                        move = x + 1, y
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x + 1, y - 1) in max_qvalue_nodes:
                        move = x + 1, y - 1
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x, y - 1) in max_qvalue_nodes:
                        move = x, y - 1
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))
                    if (x - 1, y - 1) in max_qvalue_nodes:
                        move = x - 1, y - 1
                        alphabeta = AlphaBetaPlayer()
                        value = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                        if value != "PASS":
                            heapq.heappush(value_heap, (value, move))

                    heapq._heapify_max(value_heap)
                    if value_heap:
                        placement = heapq._heappop_max(value_heap)
                        possible_placements = []
                        max_score = placement[0]
                        possible_placements.append(placement[1])
                        while value_heap:
                            placement = heapq._heappop_max(value_heap)
                            if max_score != placement[0]:
                                break
                            possible_placements.append(placement[1])
                        row, col = random.choice(possible_placements)
                        return row, col


            print("No. of moves to do alpha-beta search: ", len(max_qvalue_nodes))
            print("No. of moves: ", go.n_move)
            for move in max_qvalue_nodes:
                alphabeta = AlphaBetaPlayer()
                score = alphabeta.alpha_beta_heuristic(go, move, piece_type, start_time)
                if score != "PASS":
                    heapq.heappush(score_heap, (score, move))
            heapq._heapify_max(score_heap)
            placement = heapq._heappop_max(score_heap)
            possible_placements = []
            max_score = placement[0]
            possible_placements.append(placement[1])
            while score_heap:
                placement = heapq._heappop_max(score_heap)
                if max_score != placement[0]:
                    break
                possible_placements.append(placement[1])
            row, col = random.choice(possible_placements)
            # row, col = random.choice(max_qvalue_nodes)
            curr_time = time.time()
            print('Evaluation time: {}s'.format(round(curr_time - start_time, 7)))
            return row, col
        curr_time = time.time()
        print('Evaluation time: {}s'.format(round(curr_time - start_time, 7)))
        return "PASS"

    def get_input(self, go, piece_type, opp_move):
        """ make a move
        """
        move = self.select_best_move(go, piece_type, opp_move)
        if move != "PASS":
            self.history_states.append((go.encode_go_board(), move))
        return move


def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return piece_type, previous_board, board


def readNmoves(path="nmoves.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()
        num_moves = int(lines[0])
        return num_moves


def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def writeNmoves(num_moves, path="nmoves.txt"):
    with open(path, 'w') as f:
        f.write(str(num_moves))


def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"

    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"

    with open(path, 'w') as f:
        f.write(res[:-1])


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    num_moves = readNmoves()
    go = GO(N)
    go.set_go_board(piece_type, previous_board, board)

    opp_move = None, None
    for i in range(go.size):
        for j in range(go.size):
            if board[i][j] == 3 - piece_type:
                if previous_board[i][j] != 3 - piece_type:
                    opp_move = i, j

    if not any(piece_type in sublist for sublist in previous_board):
        num_moves = 0

    if num_moves == 0:
        if piece_type == 1:
            go.n_move = num_moves
        elif piece_type == 2:
            go.n_move = num_moves + 1
    else:
        go.n_move = num_moves + 1

    player = QLearner()
    player.q_values = player.load_qvalues_file(piece_type)
    # player = AlphaBetaPlayer()
    action = player.get_input(go, piece_type, opp_move)
    writeOutput(action)
    go.n_move += 1
    writeNmoves(go.n_move)