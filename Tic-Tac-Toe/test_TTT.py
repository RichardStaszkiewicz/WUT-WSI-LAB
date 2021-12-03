from TTT_simulator import *
import numpy as np

class TestPlayerVerify(object):
    def test_init(self):
        z = Player(-1, 3)
        assert z.depth == 3
        assert z.strategy == -1

    def test_verify_winner_MIN_high(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 1, 0]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == -1e9

    def test_verify_winner_MIN_wide(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 0, 1],
            [0, 1, 1],
            [-1, -1, -1]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == -1e9

    def test_verify_winner_MIN_across(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 0, 1],
            [0, -1, 1],
            [-1, 1, -1]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == -1e9

    def test_verify_winner_MAX_high(self):
        play = Player(-1, 3)
        state = State(np.array([
            [1, 1, -1],
            [-1, 1, -1],
            [-1, 1, 0]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == 1e9

    def test_verify_winner_MAX_wide(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 0, 1],
            [1, 1, 1],
            [-1, 1, -1]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == 1e9

    def test_verify_winner_MIN_across(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 0, 1],
            [0, 1, -1],
            [1, 1, -1]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == 1e9

    def test_verify_winner_tie(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, -1]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == True
        assert ans[1] == 0

    def test_verify_winner_still_play(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 0, 1],
            [1, 1, -1],
            [-1, 1, -1]
        ]))
        ans = play.verify_winner(state)
        assert ans[0] == False
        assert ans[1] == 0

class TestPlayerHeuristic(object):
    def test_example1(self):
        play = Player(-1, 3)
        state = State(np.array([
            [-1, 1, 1],
            [-1, 0, 1],
            [0, 1, 0]
        ]))

        assert play.heuristic(state) == -3
        assert play.single_place_hval(state, 1, 1) == 1
        assert play.single_place_hval(state, 0, 2) == 1
        assert play.single_place_hval(state, 2, 2) == 1

    def test_example2(self):
        play = Player(-1, 3)
        state = State(np.array([
            [1, -1, -1],
            [1, 0, -1],
            [0, -1, 0]
        ]), 1)

        assert play.heuristic(state) == 3
        assert play.single_place_hval(state, 1, 1) == 1
        assert play.single_place_hval(state, 0, 2) == 1
        assert play.single_place_hval(state, 2, 2) == 1

    def test_example3(self):
        play = Player(-1, 3)
        state = State(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]), 1)
        assert play.heuristic(state) == 4 * (2 + 3) + 4

        assert play.single_place_hval(state, 1, 1) == 4
        assert play.single_place_hval(state, 0, 0) == 3
        assert play.single_place_hval(state, 0, 1) == 2