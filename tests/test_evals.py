import json
import pytest
import chess
from nimzo.data.pack_evals import parse_eval_line
from nimzo.encoding import action_to_move

def test_eval_line_parsing_truncated_fen_and_chess960_castling():
    """Verify parsing of truncated FENs, highest depth selection, and Chess960 castling (e1h1 -> e1g1)."""
    # Test line with truncated FEN and Chess960 castling notation 'e1h1' (kingside castling)
    eval_json = {
        "fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -",
        "evals": [
            {"depth": 20, "pvs": [{"cp": 50, "line": "a1a8"}]},
            {"depth": 36, "pvs": [{"cp": 120, "line": "e1h1 a8a1"}]}  # Higher depth 36, line starts with e1h1 (Chess960 castling)
        ]
    }
    line_str = json.dumps(eval_json)
    
    parsed = parse_eval_line(line_str)
    assert parsed is not None
    
    board, action, cp_val, is_mate = parsed
    assert board.turn == chess.WHITE
    assert cp_val == 120
    assert not is_mate
    
    # Verify move action decodes back to valid legal move on standard board (e1g1)
    move = action_to_move(action, board)
    assert move in board.legal_moves
    assert board.is_castling(move)
    assert move.from_square == chess.E1
    assert move.to_square == chess.G1

def test_eval_line_mate_score():
    eval_json = {
        "fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5",
        "evals": [
            {"depth": 25, "pvs": [{"mate": 3, "line": "c4f7"}]}
        ]
    }
    line_str = json.dumps(eval_json)
    parsed = parse_eval_line(line_str)
    assert parsed is not None
    board, action, cp_val, is_mate = parsed
    assert is_mate == True
    assert cp_val == 3
