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


# from read import readInput
# from write import writeOutput

# from host import GO


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

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
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

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def encode_board(self):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(self.board[i][j]) for i in range(self.size) for j in range(self.size)])

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
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

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
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
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
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
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
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
        if not (0 <= i < len(board)):
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
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board

    def visualize_board(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

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
        # final_score = cnt_2 + self.komi - cnt_1
        board = self.board
        X_liberty = 0
        O_liberty = 0
        test_go = self.copy_board()
        board = test_go.board
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    board[i][j] = 1
                    test_go.update_board(board)
                    if test_go.find_liberty(i, j):
                        X_liberty += 1
                    board[i][j] = 2
                    test_go.update_board(board)
                    if test_go.find_liberty(i, j):
                        O_liberty += 1
                    board[i][j] = 0

        if piece_type == 1:
            final_score = cnt_1 + X_liberty + self.count_died_pieces[1] - cnt_2 - self.komi - O_liberty - self.count_died_pieces[2]
        elif piece_type == 2:
            final_score = cnt_2 + self.komi + O_liberty + self.count_died_pieces[2] - cnt_1 - X_liberty - self.count_died_pieces [1]

        return final_score

    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi:
            return 1
        elif cnt_1 < cnt_2 + self.komi:
            return 2
        else:
            return 0

    def play(self, player1, player2, verbose=False):
        '''
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        self.init_board(self.size)
        # Print input hints and error message if there is a manual player
        if player1.type == 'manual' or player2.type == 'manual':
            self.verbose = True
            print('----------Input "exit" to exit the program----------')
            print('X stands for black chess, O stands for white chess.')
            self.visualize_board()

        # verbose = self.verbose
        self.n_move = 0
        # Game starts!
        while 1:
            piece_type = 1 if self.X_move else 2

            # Judge if the game should end
            if self.game_end(piece_type):
                result = self.judge_winner()
                if verbose:
                    print('Game ended.')
                    if result == 0:
                        print('The game is a tie.')
                    else:
                        print('The winner is {}'.format('X' if result == 1 else 'O'))
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")

            opp_move = None, None
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i][j] == 3 - piece_type:
                        if self.previous_board[i][j] != 3 - piece_type:
                            opp_move = i, j

            # Game continues
            if piece_type == 1:
                action = player1.get_input(self, piece_type, opp_move)
            else:
                action = player2.get_input(self, piece_type, opp_move)

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board()
                    continue

                self.died_pieces = self.remove_died_pieces(3 - piece_type)  # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board()  # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move  # Players take turn


class MyPlayer():
    def __init__(self):
        self.type = 'myplayer'

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''

        '''Greedy/Aggressive'''
        # if piece_type == 1:
        score_heap = []
        for i in range(go.size):
            for j in range(go.size):
                test_go = go.copy_board()
                if test_go.place_chess(i, j, piece_type):
                    test_go.died_pieces = test_go.remove_died_pieces(3 - piece_type)
                    # test_go.visualize_board()
                    score_pos = test_go.score(piece_type)
                    score_opp = test_go.score(3 - piece_type)
                    pos = (i, j)
                    heapq.heappush(score_heap, (score_pos - score_opp, pos))
                # if go.valid_place_check(i, j, piece_type, test_check = True):
                #     possible_placements.append((i,j))
        heapq._heapify_max(score_heap)
        # print(score_heap)
        if score_heap:
            max_score_pos = heapq._heappop_max(score_heap)
            # print(max_score_pos)
            return max_score_pos[1]
        return "PASS"


class AlphaBetaPlayer:

    def __init__(self):

        # self.go = go
        # self.move = move
        # self.piece_type = piece_type
        self.type = 'alphabeta'
        # self.start_time = start_time

    # Player 'X' is max
    def max_alpha_beta(self, go, orig_piece_type, piece_type, start_time, depth, moves_chg_depth, alpha, beta):
        # board = go.board
        # num_moves = n_moves
        curr_time = time.time()
        maxv = -np.inf
        px = None
        py = None

        d = 3
        if moves_chg_depth >= 14:
            d = 4


        # if go.n_move >= go.max_move or round(curr_time - self.start_time, 7) > 9.00 or depth > d:
        # if go.n_move >= go.max_move or round(curr_time - self.start_time, 7) > 8.00:
        if go.n_move >= go.max_move or depth > d:
            # if go.n_move >= go.max_move:
            final_score = go.get_final_score(orig_piece_type)

            return (final_score, 0, 0)

        m = "PASS"
        valid_skipped_moves = []
        for i in range(0, 5):
            for j in range(0, 5):
                if go.valid_place_check(i, j, piece_type):
                    # test_go = go.copy_board()
                    # test_board = test_go.board
                    # test_board[i][j] = 3 - piece_type
                    # if not test_go.find_liberty(i, j):
                    #     valid_skipped_moves.append((i, j))
                    #     del test_go
                    #     continue

                    new_go = go.copy_board()
                    new_board = new_go.board
                    new_board[i][j] = piece_type
                    new_go.update_board(new_board)
                    new_go.died_pieces = new_go.remove_died_pieces(3 - piece_type)
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
                    # board[i][j] = 0
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
        # board = go.board
        # num_moves = n_moves
        curr_time = time.time()
        minv = np.inf
        qx = None
        qy = None

        d = 3
        if moves_chg_depth >= 14:
            d = 4


        # if go.n_move >= go.max_move or round(curr_time - self.start_time, 7) > 9.00 or depth > d:
        # if go.n_move >= go.max_move or round(curr_time - self.start_time, 7) > 8.00:
        if go.n_move >= go.max_move or depth > d:
            # if go.n_move >= go.max_move:
            final_score = go.get_final_score(orig_piece_type)
            # print(depth)
            return (final_score, 0, 0)


        m = "PASS"
        valid_skipped_moves = []
        for i in range(0, 5):
            for j in range(0, 5):
                if go.valid_place_check(i, j, piece_type):
                    # test_go = go.copy_board()
                    # test_board = test_go.board
                    # test_board[i][j] = 3 - piece_type
                    # if not test_go.find_liberty(i, j):
                    #     valid_skipped_moves.append((i, j))
                    #     del test_go
                    #     continue

                    new_go = go.copy_board()
                    new_board = new_go.board
                    new_board[i][j] = piece_type
                    new_go.update_board(new_board)
                    new_go.died_pieces = new_go.remove_died_pieces(3 - piece_type)
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
                    # board[i][j] = 0
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
            if board[2][2] == 0 and go.valid_place_check(2, 2, piece_type):
                return 2, 2
            if board[1][1] == 0 and go.valid_place_check(1, 1, piece_type):
                return 1, 1
            if board[1][3] == 0 and go.valid_place_check(1, 3, piece_type):
                return 1, 3
            if board[3][1] == 0 and go.valid_place_check(3, 1, piece_type):
                return 3, 1
            if board[3][3] == 0 and go.valid_place_check(3, 3, piece_type):
                return 3, 3
            if (x, y) != (None, None):
                score_heap = []
                if go.valid_place_check(x - 1, y, piece_type):
                    move = x - 1, y
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x - 1, y + 1, piece_type):
                    move = x - 1, y + 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x, y + 1, piece_type):
                    move = x, y + 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x + 1, y + 1, piece_type):
                    move = x + 1, y + 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x + 1, y, piece_type):
                    move = x + 1, y
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x + 1, y - 1, piece_type):
                    move = x + 1, y - 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x, y - 1, piece_type):
                    move = x, y - 1
                    m = self.alpha_beta_heuristic(go, move, piece_type, start_time)
                    if m != "PASS":
                        heapq.heappush(score_heap, (m, move))
                if go.valid_place_check(x - 1, y - 1, piece_type):
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

        test_go = go.copy_board()

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

        test_go = go.copy_board()
        test_board = test_go.board
        x, y = move
        test_board[x][y] = piece_type
        test_go.update_board(test_board)
        test_go.died_pieces = test_go.remove_died_pieces(3 - piece_type)
        test_go.n_move += 1
        moves_chg_depth = test_go.n_move
        new_piece_type = 3 - piece_type
        orig_piece_type = piece_type
        depth = 1
        (m, i, j) = self.max_alpha_beta(test_go, orig_piece_type, new_piece_type, start_time, depth,
                                        moves_chg_depth, -np.inf, np.inf)
        return m


class QLearner:
    GAME_NUM = 2

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

    def set_side(self, side):
        self.side = side

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
            # q_val = np.zeros((go.size, go.size))
            q_val = [[self.initial_value for i in range(go.size)] for j in range(go.size)]
            # q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def select_best_move(self, go, piece_type, opp_move):
        start_time = time.time()
        state = go.encode_board()
        # self.q_values = self.load_qvalues_file(piece_type)
        # if state not in self.q_values:
        #     state_present = False
        # else:
        #     state_present = True
        q_values = self.Q(go, state)
        curr_max = -np.inf
        max_qvalue_nodes = []
        board = go.board
        for i in range(go.size):
            for j in range(go.size):
                # if not state_present:
                # if not go.valid_place_check(i, j, piece_type):
                #     q_values[i][j] = -1.0
                if go.valid_place_check(i, j, piece_type):
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
        # if piece_type == 1:
        #     move = self._select_best_move_black(go, piece_type)
        # elif piece_type == 2:
        #     move = self._select_best_move_white(go, piece_type)
        # if move != "PASS":
        #     self.history_states.append((go.encode_board(), move))
        move = self.select_best_move(go, piece_type, opp_move)
        if move != "PASS":
            self.history_states.append((go.encode_board(), move))
        return move

    def get_max_q_value(self, q):
        max_value = -np.inf
        for l in q:
            max_in_list = max(l)
            if max_value < max_in_list:
                max_value = max_in_list
        return max_value

    def learn(self, result, go, piece_type):
        """ when games ended, this method will be called to update the qvalues
        """
        if result == 0:
            reward = DRAW_REWARD
        elif result == piece_type:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            q = self.Q(go, state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value

            max_q_value = np.max(q)
            self.q_values[state] = q
        self.history_states = []

        json_object = json.dumps(self.q_values, indent=4)
        # print(json_object)
        if piece_type == 1:
            with open("qvalues_black.json", "w") as outfile:
                # json.dump(self.q_values, outfile)
                outfile.write(json_object)
        elif piece_type == 2:
            with open("qvalues_white.json", "w") as outfile:
                # json.dump(self.q_values, outfile)
                outfile.write(json_object)


class RandomPlayer():
    def __init__(self):
        self.type = 'random'

    def get_input(self, go, piece_type, opp_move):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    possible_placements.append((i, j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)


def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return piece_type, previous_board, board


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
        f.write(res[:-1]);


def greedy_player():
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)

    action, x, y = readOutput()
    if action == "MOVE":
        if not go.place_chess(x, y, piece_type):
            print('Game end.')
            print('The winner is {}'.format('X' if 3 - piece_type == 1 else 'O'))
            sys.exit(3 - piece_type)

        go.died_pieces = go.remove_died_pieces(3 - piece_type)

    go.visualize_board()

    piece_type = 2 if piece_type == 1 else 1
    if action == "PASS":
        go.previous_board = go.board
    writeNextInput(piece_type, go.previous_board, go.board)


def random_player():
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player1 = MyPlayer()
    player2 = RandomPlayer()
    if piece_type == 1:
        go.play(player1, player2, verbose=True)
    else:
        go.play(player2, player1, verbose=True)
    # go.visualize_board()


def qlearner_player():
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)

    opp_move = None, None
    for i in range(go.size):
        for j in range(go.size):
            if board[i][j] == 3 - piece_type:
                if previous_board[i][j] != 3 - piece_type:
                    opp_move = i, j

    ply_qlearner = QLearner()
    ply_qlearner2 = QLearner()
    # ply_random = RandomPlayer()
    # ply_alphabeta = AlphaBetaPlayer()

    final_result = {}
    final_result['QPlayer'] = ''
    final_result['X'] = 0
    final_result['O'] = 0
    final_result['Draw'] = 0
    final_result['Total'] = 0

    for i in range(50):
        print('Game No.', i + 1)
        start_time = time.time()
        ply_qlearner.q_values = ply_qlearner.load_qvalues_file(piece_type)
        ply_qlearner2.q_values = ply_qlearner2.load_qvalues_file(3 - piece_type)
        if piece_type == 1:
            result = go.play(ply_qlearner, ply_qlearner2, verbose=False)
            # result = go.play(ply_qlearner, ply_random, verbose=False)
            # result = go.play(ply_qlearner, ply_alphabeta, verbose=False)
            # result = go.play(ply_random, ply_alphabeta, verbose=False)
        else:
            result = go.play(ply_qlearner2, ply_qlearner, verbose=False)
            # result = go.play(ply_random, ply_qlearner, verbose=False)
            # result = go.play(ply_alphabeta, ply_qlearner, verbose=False)
            # result = go.play(ply_alphabeta, ply_random, verbose=False)
        # result = go.play(player1, player2)
        ply_qlearner.learn(result, go, piece_type)
        ply_qlearner2.learn(result, go, 3 - piece_type)
        end_time = time.time()
        print('Evaluation time: {}s'.format(round(end_time - start_time, 7)))
        go.visualize_board()
        if result == 1:
            final_result['X'] += 1
        elif result == 2:
            final_result['O'] += 1
        elif result == 0:
            final_result['Draw'] += 1
        final_result['Total'] += 1
        if piece_type == 1:
            final_result['QPlayer'] = 'X'
        else:
            final_result['QPlayer'] = 'O'
        print(final_result)
        print('total states captured:', len(ply_qlearner.q_values.keys()))



    final_result = {}
    final_result['QPlayer'] = ''
    final_result['X'] = 0
    final_result['O'] = 0
    final_result['Draw'] = 0
    final_result['Total'] = 0

    piece_type = 3 - piece_type

    for i in range(50):
        print('Game No.', i + 1)
        start_time = time.time()
        ply_qlearner.q_values = ply_qlearner.load_qvalues_file(piece_type)
        ply_qlearner2.q_values = ply_qlearner2.load_qvalues_file(3 - piece_type)
        if piece_type == 1:
            result = go.play(ply_qlearner, ply_qlearner2, verbose=False)
            # result = go.play(ply_qlearner, ply_random, verbose=False)
            # result = go.play(ply_qlearner, ply_alphabeta, verbose=False)
            # result = go.play(ply_random, ply_alphabeta, verbose=False)
        else:
            result = go.play(ply_qlearner2, ply_qlearner, verbose=False)
            # result = go.play(ply_random, ply_qlearner, verbose=False)
            # result = go.play(ply_alphabeta, ply_qlearner, verbose=False)
            # result = go.play(ply_alphabeta, ply_random, verbose=False)
        # result = go.play(player1, player2)
        ply_qlearner.learn(result, go, piece_type)
        ply_qlearner2.learn(result, go, 3 - piece_type)
        end_time = time.time()
        print('Evaluation time: {}s'.format(round(end_time - start_time, 7)))
        go.visualize_board()
        if result == 1:
            final_result['X'] += 1
        elif result == 2:
            final_result['O'] += 1
        elif result == 0:
            final_result['Draw'] += 1
        final_result['Total'] += 1
        if piece_type == 1:
            final_result['QPlayer'] = 'X'
        else:
            final_result['QPlayer'] = 'O'
        print(final_result)
        print('total states captured:', len(ply_qlearner.q_values.keys()))


if __name__ == "__main__":
    # greedy_player()
    # random_player()
    qlearner_player()
