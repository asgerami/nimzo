import sys
import argparse
import chess
from nimzo.model import build_model_from_config
from nimzo.engine import NimzoEngine

def play_cli_game(config_path: str = None, weights_path: str = None, playouts: int = 400):
    if not config_path or not weights_path:
        print("Error: --config and --weights are required. There is no fallback mode; "
              "Nimzo only plays with a trained checkpoint loaded.")
        sys.exit(1)

    import torch
    model = build_model_from_config(config_path)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded Nimzo ResNet model successfully.")

    engine = NimzoEngine(model=model, use_mcts=True, playouts=playouts)
    board = chess.Board()
    history = []

    print("\n" + "="*40)
    print("      Nimzo — Terminal Chess Engine")
    print("="*40)
    print("Commands: Enter move in SAN or UCI (e.g. e4, Nf3, e2e4).")
    print("          Type 'undo' to revert last move pair.")
    print("          Type 'quit' or 'exit' to end game.\n")
    
    while not board.is_game_over():
        print(board)
        print("-" * 25)
        
        if board.turn == chess.WHITE:
            user_input = input("\nYour move (White): ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Game exited.")
                break
            if user_input.lower() == "undo":
                if len(history) >= 2:
                    board.pop()
                    board.pop()
                    history.pop()
                    history.pop()
                    print("Undid last 2 moves.")
                elif len(history) == 1:
                    board.pop()
                    history.pop()
                    print("Undid last move.")
                else:
                    print("No moves to undo.")
                continue
                
            try:
                move = board.parse_san(user_input) if not user_input.isalnum() or len(user_input) < 4 else chess.Move.from_uci(user_input)
                if move not in board.legal_moves:
                    print("Illegal move! Try again.")
                    continue
                board.push(move)
                history.append(move)
            except Exception as e:
                print(f"Invalid input '{user_input}' ({e}). Example valid input: 'e4' or 'e2e4'.")
                continue
        else:
            print("\nEngine (Black) is calculating move...")
            mcts_move = engine.get_best_move(board, movetime_ms=1000)
            print(f"Engine plays: {mcts_move.uci()} ({board.san(mcts_move)})")
            board.push(mcts_move)
            history.append(mcts_move)
            
    if board.is_game_over():
        print("\n" + "="*40)
        print(board)
        print("="*40)
        print(f"Game Over! Result: {board.result()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nimzo Terminal Chess Interface")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--playouts", type=int, default=400, help="MCTS playouts per move")
    args = parser.parse_args()
    play_cli_game(args.config, args.weights, playouts=args.playouts)
