from TTT_simulator import *
import numpy as np

class TestStateVerify(object):
    def test_init(self):
        z = Player(-1, 3)
        assert z.depth == 3
        assert z.strategy == -1

    def test_str(self):
        x = State(np.array([
            [-1, 0, 1],
            [0, 1, -1],
            [0, 1, -1]
        ])).__str__()
        assert x == "O |   | X\n---------\n  | X | O\n---------\n  | X | O"

    def test_verify_winner_MIN_high(self):
        state = State(np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 1, 0]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == -1e9

    def test_verify_winner_MIN_wide(self):
        state = State(np.array([
            [-1, 0, 1],
            [0, 1, 1],
            [-1, -1, -1]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == -1e9

    def test_verify_winner_MIN_across(self):
        state = State(np.array([
            [-1, 0, 1],
            [0, -1, 1],
            [-1, 1, -1]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == -1e9

    def test_verify_winner_MAX_high(self):
        state = State(np.array([
            [1, 1, -1],
            [-1, 1, -1],
            [-1, 1, 0]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == 1e9

    def test_verify_winner_MAX_wide(self):
        state = State(np.array([
            [-1, 0, 1],
            [1, 1, 1],
            [-1, 1, -1]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == 1e9

    def test_verify_winner_MIN_across(self):
        state = State(np.array([
            [-1, 0, 1],
            [0, 1, -1],
            [1, 1, -1]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == 1e9

    def test_verify_winner_tie(self):
        state = State(np.array([
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, -1]
        ]))
        ans = state.verify_winner()
        assert ans[0] == True
        assert ans[1] == 0

    def test_verify_winner_still_play(self):
        state = State(np.array([
            [-1, 0, 1],
            [1, 1, -1],
            [-1, 1, -1]
        ]))
        ans = state.verify_winner()
        assert ans[0] == False
        assert ans[1] == 0


class TestStateHeuristic(object):
    def test_example1(self):
        state = State(np.array([
            [-1, 1, 1],
            [-1, 0, 1],
            [0, 1, 0]
        ]))

        assert state.heuristic() == -3
        assert state.single_place_hval(1, 1) == 1
        assert state.single_place_hval(0, 2) == 1
        assert state.single_place_hval(2, 2) == 1

    def test_example2(self):
        state = State(np.array([
            [1, -1, -1],
            [1, 0, -1],
            [0, -1, 0]
        ]), 1)

        assert state.heuristic() == 3
        assert state.single_place_hval(1, 1) == 1
        assert state.single_place_hval(0, 2) == 1
        assert state.single_place_hval(2, 2) == 1

    def test_example3(self):
        state = State(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]), 1)
        assert state.heuristic() == 4 * (2 + 3) + 4

        assert state.single_place_hval(1, 1) == 4
        assert state.single_place_hval(0, 0) == 3
        assert state.single_place_hval(0, 1) == 2