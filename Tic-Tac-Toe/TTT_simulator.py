import numpy as np
from copy import deepcopy

INF = 1e9


class State(object):
    def __init__(self, board=None, move=None) -> None:
        if(board is not None): #list [3][3] => -1 MIN, 1 MAX, 0 spare
            self.board = board
        else:
            self.board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.move = move #1 -> MAX, -1 -> MIN

    def verify_winner(self):
        if(3 in sum(self.board) or 3 in sum(self.board.T)): return (True, INF) # MAX has won
        if(-3 in sum(self.board) or -3 in sum(self.board.T)): return (True, -INF) #MIN has won

        auxiliary_sum_right = 0
        auxiliary_sum_left = 0
        for i in (0, 1, 2):
            auxiliary_sum_right += self.board[i][i]
            auxiliary_sum_left += self.board[i][2-i]
        if(auxiliary_sum_left == 3 or auxiliary_sum_right == 3): return (True, INF) #MAX has won
        if(auxiliary_sum_left == -3 or auxiliary_sum_right == -3): return (True, -INF) #MIN has won

        if(0 in self.board): return (False, 0) # there are still possible moves
        return (True, 0) # there are no possible moves, nobody has won

    def heuristic(self):
        hvalue = 0
        for y in (0, 1, 2):
            for x in (0, 1, 2):
                if self.board[y][x] == 0:
                    hvalue += self.single_place_hval(x, y)
        return hvalue * self.move * -1

    def single_place_hval(self, x, y):
        hvalue = 0
        if self.move * -1 not in self.board[y]: hvalue += 1
        if self.move * -1 not in self.board.T[x]: hvalue += 1

        if(x == y):
            if self.move * -1 not in [self.board[i][i] for i in (0, 1, 2)]: hvalue += 1
        if(y == 2-x):
            if self.move * -1 not in [self.board[2-i][i] for i in (0, 1, 2)]: hvalue += 1

        return hvalue

    def __str__(self):
        # X -> MAX
        # O -> MIN
        # _ -> not taken
        info = ""
        for y in (0, 1, 2):
            for x in (0, 1, 2):
                if self.board[y][x] == 1:
                    info += "X"
                elif self.board[y][x] == -1:
                    info += "O"
                else:
                    info += " "
                if x != 2: info += " | "

            if y != 2: info += "\n---------\n"

        return info



class Game(object):
    def __init__(self, state = None) -> None:
        if state is not None: self.state = state
        else: self.state = State()

    def exe(self, Player1, Player2):
        if Player1.strategy + Player2.strategy != 0:
            print("Players must be MIN-MAX")
            return Exception("Player error")

        if self.state.move is None:
            self.state.move = Player1.strategy

        in_game = self.state.verify_winner()
        print(self)
        while(in_game[0] == False):
            self.state = Player1.move(self.state)
            print(self)
            in_game = self.state.verify_winner()
            if in_game[0]: break
            self.state = Player2.move(self.state)
            print(self)
            in_game = self.state.verify_winner()

        if in_game[1] == 0:
            print("It's a tie!\n")
        elif in_game[1] > 0:
            print("MAX has won!\n")
        elif in_game[1] < 0:
            print("MIN has won!\n")

        return 0

    def __str__(self):
        info = ""
        if self.state.move == -1:
            x = "MIN move"
        else:
            x = "MAX move"
        info += f"\n----{x}----\n"
        info += self.state.__str__()
        return info


class Player(object):
    def __init__(self, type, depth) -> None:
        self.strategy = type # 1 -> MAX, -1 -> MIN
        self.depth = depth # iteration depth


    def move(self, gamestate: State()):
        return self.make_move(gamestate, self.depth)[1] #return the last upper propagated state


    def make_move(self, gamestate: State(), depth: int()):
        win = gamestate.verify_winner()
        if win[0]:
            return (win[1], gamestate)

        if depth == 0:
            return (gamestate.heuristic(), gamestate)
        else: depth -= 1


        options = []
        for x in range(3):
            for y in range(3):
                if gamestate.board[x][y] == 0:
                    copy_gamestate = deepcopy(gamestate)
                    copy_gamestate.board[x][y] = gamestate.move
                    copy_gamestate.move *= -1
                    options.append(self.make_move(copy_gamestate, depth))

        options = sorted(options, key=lambda x: x[0]) # sort from min to max via payback

        if depth == self.depth - 2:
            if gamestate.move == 1: return (options[-1][0], gamestate) # if MAX, return maximal payback for current gamestate
            else: return (options[0][0], gamestate) # if MIN, return minimal payback for current gamestate

        if gamestate.move == 1: return (options[-1][0], options[-1][1]) # if MAX, return maximal payback for current gamestate
        else: return (options[0][0], options[0][1]) # if MIN, return minimal payback for current gamestate


# Obsługa
# 1. Zakładamy obiekt Game
# 1.1 Jeżeli chcemy zacząć z określonego punktu, w konstruktorze podajemy paramentr typu State
# 2. Zakładamy dwóch garczy, MIN i MAX
# 2.1 Gracz MIN jako pierwszy argument przyjmuje -1, a gracz MAX 1
# 2.2 Drugim parametrem jest głębokość schodzenia algorytmu >= 1
# 3. Wywołujemy metodę Game.exe(P1, P2)
# 3.1 Zaczyna gracz wg parametrów zadeklarowanego stanie gry
# 3.2 Zaczyna gracz, podany w metodzie jako pierwszy


if __name__ == "__main__":
    # G1 = Game(State(np.array([
    #     [1, -1, -1],
    #     [0, 0, 0],
    #     [-1, 1, -1]
    # ])))
    G1 = Game()
    G1.exe(Player(-1, 1), Player(1, 9))
