import sys
import subprocess
import pytest
import torch
import chess
from nimzo.model import ChessResNet, build_model_from_config
from nimzo.search.mcts import MCTS, MCTSNode
from nimzo.engine import NimzoEngine

def test_terminal_detection_catches_fifty_move_and_threefold():
    """MCTS must recognize fifty-move-rule and threefold-repetition draws as
    terminal (exact 0.0) without ever evaluating them through the network --
    the default is_game_over(claim_draw=False) misses both, which was a real
    bug (positions fell through to NN evaluation instead of an exact score)."""

    class ExplodingModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("terminal position must not reach the network")
        def eval(self):
            return self

    mcts = MCTS(model=ExplodingModel())

    # Fifty-move rule: halfmove clock at 100.
    fifty_move_board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 100 60")
    assert fifty_move_board.is_game_over(claim_draw=True)

    root = MCTSNode()
    mcts._evaluate_and_backprop_batch([([root], fifty_move_board)])
    assert root.is_terminal
    assert root.terminal_value == 0.0

    # Threefold repetition: two full knight-shuffle cycles back to the start.
    threefold_board = chess.Board()
    for _ in range(2):
        for san in ["Nf3", "Nf6", "Ng1", "Ng8"]:
            threefold_board.push_san(san)
    assert threefold_board.is_game_over(claim_draw=True)

    root2 = MCTSNode()
    mcts._evaluate_and_backprop_batch([([root2], threefold_board)])
    assert root2.is_terminal
    assert root2.terminal_value == 0.0

def test_mcts_zero_illegal_moves_and_playouts_sec_gate():
    """Milestone M6 Gate: MCTS generates 100% legal moves and measures >200 playouts/sec."""
    model = ChessResNet(num_blocks=2, channels=32)
    engine = NimzoEngine(model=model, use_mcts=True, playouts=200)
    
    board = chess.Board()
    moves_played = 0
    
    import time
    t0 = time.time()
    
    # Run 20 self-play moves
    while not board.is_game_over() and moves_played < 20:
        move = engine.get_best_move(board, movetime_ms=500)
        assert move in board.legal_moves
        board.push(move)
        moves_played += 1
        
    elapsed = time.time() - t0
    total_playouts = moves_played * 200
    playouts_per_sec = total_playouts / elapsed if elapsed > 0 else 0
    
    print(f"\n[M6 Gate Result] Total Moves: {moves_played}, Time: {elapsed:.2f}s, Measured Playouts/sec: {playouts_per_sec:.1f}")
    assert playouts_per_sec >= 200.0

def test_uci_handshake(tmp_path):
    """Milestone M6 Gate: Verify UCI protocol handshake ('uci' -> 'uciok', 'isready' -> 'readyok')."""
    ckpt_path = tmp_path / "handshake_ckpt.pt"
    model = build_model_from_config("configs/tiny.yaml")
    torch.save({"model": model.state_dict()}, ckpt_path)

    process = subprocess.Popen(
        [sys.executable, "-m", "nimzo.uci", "--config", "configs/tiny.yaml", "--weights", str(ckpt_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, _ = process.communicate(input="uci\nisready\nquit\n", timeout=5)
    assert "uciok" in stdout
    assert "readyok" in stdout

def test_uci_requires_model():
    """Nimzo has no random-move fallback: no --config/--weights must exit cleanly, not hang or crash."""
    process = subprocess.Popen(
        [sys.executable, "-m", "nimzo.uci"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate(input="uci\n", timeout=5)
    assert process.returncode != 0
    assert "required" in stderr.lower()
