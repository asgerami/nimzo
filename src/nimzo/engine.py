import chess
from typing import Optional
from nimzo.model import ChessResNet
from nimzo.search.policy import best_move_policy
from nimzo.search.mcts import MCTS

class NimzoEngine:
    def __init__(self, model: Optional[ChessResNet] = None, use_mcts: bool = True, playouts: int = 800, c_puct: float = 1.5):
        if model is None:
            raise ValueError(
                "NimzoEngine requires a loaded model; there is no random-move fallback. "
                "Pass --config and --weights pointing at a trained checkpoint."
            )
        self.model = model
        self.use_mcts = use_mcts
        self.playouts = playouts
        self.mcts = MCTS(model=model, c_puct=c_puct) if use_mcts else None
        # Position the tree was last searched from, so get_best_move can tell
        # whether the incoming board is a pure extension (normal play) vs. a
        # divergence (takeback, new game, arbitrary FEN jump).
        self._searched_board: Optional[chess.Board] = None

    def get_best_move(self, board: chess.Board, movetime_ms: int = 1000, nodes: Optional[int] = None) -> chess.Move:
        if not list(board.legal_moves):
            raise ValueError("No legal moves available")

        if self.use_mcts:
            self._sync_tree(board)
            if nodes is not None:
                # Node-limited search (e.g. cutechess `go nodes N`): playouts
                # is the sole stopping condition, no wall-clock cutoff. This
                # must be honored precisely for node-based Elo matches to be
                # a fair, comparable measurement against Stockfish.
                move = self.mcts.search(board, playouts=nodes, time_limit_sec=0)
            else:
                time_limit_sec = movetime_ms / 1000.0 if movetime_ms > 0 else 1.0
                move = self.mcts.search(board, playouts=self.playouts, time_limit_sec=time_limit_sec)
            self._searched_board = board.copy()
            return move
        else:
            return best_move_policy(self.model, board)

    def _sync_tree(self, board: chess.Board) -> None:
        """Advance the MCTS tree to match `board` if it's a pure extension of
        the position last searched from, so playout statistics compound across
        a game instead of restarting from scratch every move. Falls back to a
        fresh tree (mcts.root = None) on any divergence."""
        prev = self._searched_board
        if prev is None or self.mcts.root is None:
            return

        prev_moves = prev.move_stack
        new_moves = board.move_stack
        if len(new_moves) <= len(prev_moves) or new_moves[:len(prev_moves)] != prev_moves:
            self.mcts.root = None
            return

        replay_board = prev.copy()
        for move in new_moves[len(prev_moves):]:
            self.mcts.advance_root(replay_board, move)
            replay_board.push(move)
