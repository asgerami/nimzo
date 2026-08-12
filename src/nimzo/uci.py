import sys
import chess
import argparse
from nimzo.model import build_model_from_config
from nimzo.engine import NimzoEngine

def run_uci(config_path: str = None, weights_path: str = None):
    board = chess.Board()

    if not config_path or not weights_path:
        sys.stderr.write("Error: --config and --weights are required; there is no fallback mode.\n")
        sys.exit(1)

    import torch
    model = build_model_from_config(config_path)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    engine = NimzoEngine(model=model, use_mcts=True, playouts=400)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
                
            tokens = line.split()
            cmd = tokens[0]
            
            if cmd == "uci":
                print("id name Nimzo")
                print("id author Nimzo team")
                print("uciok")
                sys.stdout.flush()
            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()
            elif cmd == "ucinewgame":
                board = chess.Board()
                engine = NimzoEngine(model=model, use_mcts=True, playouts=400)
            elif cmd == "position":
                if len(tokens) > 1 and tokens[1] == "startpos":
                    board = chess.Board()
                    moves_idx = 2
                elif len(tokens) > 2 and tokens[1] == "fen":
                    fen_parts = []
                    idx = 2
                    while idx < len(tokens) and tokens[idx] != "moves":
                        fen_parts.append(tokens[idx])
                        idx += 1
                    fen_str = " ".join(fen_parts)
                    board = chess.Board(fen_str)
                    moves_idx = idx
                else:
                    moves_idx = len(tokens)
                    
                if moves_idx < len(tokens) and tokens[moves_idx] == "moves":
                    for m_str in tokens[moves_idx + 1:]:
                        board.push_uci(m_str)
            elif cmd == "go":
                nodes = None
                if "nodes" in tokens:
                    idx = tokens.index("nodes")
                    if idx + 1 < len(tokens):
                        nodes = int(tokens[idx + 1])

                movetime = 1000
                if "movetime" in tokens:
                    idx = tokens.index("movetime")
                    if idx + 1 < len(tokens):
                        movetime = int(tokens[idx + 1])

                if nodes is not None:
                    best_m = engine.get_best_move(board, nodes=nodes)
                else:
                    best_m = engine.get_best_move(board, movetime_ms=movetime)
                print(f"bestmove {best_m.uci()}")
                sys.stdout.flush()
            elif cmd in ["quit", "stop"]:
                break
        except Exception as e:
            sys.stderr.write(f"UCI loop error: {e}\n")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nimzo UCI interface")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights checkpoint")
    args = parser.parse_args()
    run_uci(args.config, args.weights)
