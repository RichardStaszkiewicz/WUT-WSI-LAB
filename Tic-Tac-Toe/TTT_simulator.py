INF = 1e9


class State(object):
    def __init__(self, board, move) -> None:
        self.board = board #list [3][3] => -1 MIN, 1 MAX, 0 spare
        self.move = move #1 -> MAX, -1 -> MIN

class Game(object):
    def __init__(self, state) -> None:
        self.state = state

class Player(object):
    def __init__(self, type, depth) -> None:
        self.strategy = type # 1 -> MAX, -1 -> MIN
        self.depth = depth # iteration depth


    def move(self, gamestate: State()):
        return self.make_move(gamestate, self.depth)[1] #return the last upper propagated state


    def make_move(self, gamestate: State(), depth: int()):
        if depth == 0:
            win = self.verify_winner(gamestate)
            if win[0]:
                return (win[1], gamestate)
            else:
                return (self.heuristic(gamestate), gamestate)
        else: depth -= 1

        options = []
        for x in range(3):
            for y in range(3):
                if gamestate.board[x][y] == 0:
                    copy_gamestate = gamestate
                    copy_gamestate.board[x][y] = gamestate.move
                    copy_gamestate.move *= -1
                    options.append(self.make_move(copy_gamestate, depth))

        options = sorted(options, key=lambda x: x[1]) # sort from min to max via payback

        if gamestate.move == 1: return (options[-1][0], gamestate) # if MAX, return maximal payback for current gamestate
        else: return (options[0][0], gamestate) # if MIN, return minimal payback for current gamestate


    def verify_winner(gamestate: State()):
        conclusion = True
        winner_payback = -INF
        return (conclusion, winner_payback)


    def heuristic(gamestate: State()):
        pass




if __name__ == "__main__":
    pass