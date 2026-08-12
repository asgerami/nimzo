import pytest
import chess
from nimzo.baselines import random_legal, greedy_material, minimax_psqt

def test_minimax_vs_random_match():
    """Milestone M1 Gate: minimax_psqt beats random_legal across games."""
    minimax_wins = 0
    random_wins = 0
    draws = 0
    num_games = 50
    
    print(f"\nRunning M1 Gate Match: minimax_psqt (depth 3) vs random_legal ({num_games} games)...")
    
    for game_idx in range(num_games):
        board = chess.Board()
        # Alternate colors
        minimax_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        move_count = 0
        
        while not board.is_game_over() and move_count < 100:
            if board.turn == minimax_color:
                move = minimax_psqt(board, depth=3)
            else:
                move = random_legal(board)
            board.push(move)
            move_count += 1
            
        result = board.result()
        if result == "1-0":
            if minimax_color == chess.WHITE:
                minimax_wins += 1
            else:
                random_wins += 1
        elif result == "0-1":
            if minimax_color == chess.BLACK:
                minimax_wins += 1
            else:
                random_wins += 1
        else:
            draws += 1
            
    print(f"Match Results: minimax_psqt {minimax_wins} wins, random_legal {random_wins} wins, {draws} draws.")
    assert minimax_wins > random_wins
    assert random_wins == 0  # Random legal should never win against 3-ply minimax
