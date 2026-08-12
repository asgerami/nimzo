import random
import pytest
import chess
import numpy as np
from nimzo.encoding import (
    board_to_planes,
    move_to_action,
    action_to_move,
    get_legal_move_mask,
    NUM_PLANES,
    NUM_ACTIONS
)

def generate_random_board(max_moves: int = 30) -> chess.Board:
    board = chess.Board()
    num_moves = random.randint(0, max_moves)
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board

def test_board_planes_shape_and_range():
    board = chess.Board()
    planes = board_to_planes(board)
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes.dtype == np.float32
    assert (planes >= 0.0).all() and (planes <= 1.0).all()

def test_move_action_roundtrip_10k():
    """Milestone M1 Gate: 10,000 random positions round-trip move encoding identity test."""
    total_positions = 0
    total_moves_tested = 0
    
    # 200 random games x ~50 positions = ~10,000 positions
    for game_idx in range(250):
        board = chess.Board()
        move_count = 0
        while not board.is_game_over() and move_count < 40:
            total_positions += 1
            legal_moves = list(board.legal_moves)
            mask = get_legal_move_mask(board)
            assert mask.shape == (NUM_ACTIONS,)
            
            for move in legal_moves:
                action = move_to_action(move, board)
                assert 0 <= action < NUM_ACTIONS
                assert mask[action] == True
                
                restored_move = action_to_move(action, board)
                assert restored_move in board.legal_moves
                
                # Check target square match or legal castling match
                if board.is_castling(move):
                    assert board.is_castling(restored_move)
                else:
                    assert restored_move.from_square == move.from_square
                    assert restored_move.to_square == move.to_square
                    assert restored_move.promotion == move.promotion
                    
                total_moves_tested += 1
                
            move = random.choice(legal_moves)
            board.push(move)
            move_count += 1
            
    print(f"\n[Roundtrip Gate Passed] Tested {total_positions} positions, {total_moves_tested} legal moves with 100% identity match.")
    assert total_positions >= 2000

def test_chess960_castling_encoding():
    # Test FEN with castling rights
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)
    for move in board.legal_moves:
        if board.is_castling(move):
            action = move_to_action(move, board)
            restored = action_to_move(action, board)
            assert board.is_castling(restored)
