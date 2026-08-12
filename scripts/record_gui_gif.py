#!/usr/bin/env python3
"""
record_gui_gif.py — Record a crisp, pixel-perfect, sped-up animated GIF of the Nimzo GUI playing chess.
Suitable for social media posts (LinkedIn, Telegram, X/Twitter, GitHub README, etc.).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import chess
import chess.pgn
import customtkinter as ctk
import torch
from PIL import ImageGrab

from nimzo.engine import NimzoEngine
from nimzo.gui import NimzoGUI, build_model_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Record a GIF of Nimzo GUI playing chess.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, default="runs/base_v4/ckpt_150000.pt", help="Path to weights")
    parser.add_argument("--output", type=str, default="results/nimzo_gui_play.gif", help="Output GIF path")
    parser.add_argument("--playouts", type=int, default=1600, help="MCTS playouts per move (1600 = Max Difficulty)")
    parser.add_argument("--max-moves", type=int, default=44, help="Maximum half-moves (ply) to play")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for move playback")
    parser.add_argument("--hold-end-sec", type=float, default=3.5, help="Seconds to hold final board state")
    parser.add_argument("--canvas-size", type=int, default=560, help="Board canvas size in pixels")
    parser.add_argument("--pgn", type=str, default=None, help="Optional PGN file to replay instead of engine self-play")
    return parser.parse_args()


def record_game_gif(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="nimzo_gif_frames_")
    print(f"[*] Temporary frame directory: {temp_dir}")

    print(f"[*] Loading model from {args.weights}...")
    model = build_model_from_config(args.config)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    engine = NimzoEngine(model=model, use_mcts=True, playouts=args.playouts)

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = NimzoGUI(root, config_path=args.config, weights_path=args.weights, canvas_size=args.canvas_size)

    root.update_idletasks()
    root.update()
    time.sleep(0.5)
    root.update()

    # Get precise window placement for full-screen grab & crop
    rx = root.winfo_rootx()
    ry = root.winfo_rooty()
    crop_w = 944
    crop_h = 690
    print(f"[*] Window position: rx={rx}, ry={ry}. Target crop: {crop_w}x{crop_h}")

    frame_counter = 0

    def capture_frame():
        nonlocal frame_counter
        root.update_idletasks()
        root.update()
        full_img = ImageGrab.grab()
        cropped = full_img.crop((rx, ry, rx + crop_w, ry + crop_h))
        frame_path = os.path.join(temp_dir, f"frame_{frame_counter:04d}.png")
        cropped.save(frame_path)
        frame_counter += 1

    # Initial frame
    capture_frame()

    moves_to_play = []
    if args.pgn:
        print(f"[*] Replaying PGN from {args.pgn}...")
        with open(args.pgn) as f:
            game = chess.pgn.read_game(f)
            moves_to_play = list(game.mainline_moves())
    
    ply = 0
    print(f"[*] Starting chess game recording (max_moves={args.max_moves})...")

    while not app.board.is_game_over() and ply < args.max_moves:
        ply += 1
        current_turn = "White" if app.board.turn == chess.WHITE else "Black"

        if moves_to_play:
            if ply - 1 >= len(moves_to_play):
                break
            move = moves_to_play[ply - 1]
            print(f"  Move {ply}: PGN move ({current_turn}) -> {move}")
        else:
            print(f"  Move {ply}: Nimzo thinking ({current_turn})...", end="", flush=True)
            move = engine.get_best_move(app.board, movetime_ms=500)
            print(f" played {move}")

        # 1. Visual selection frame (shows selected square in yellow and candidate target dots)
        from_sq = move.from_square
        to_sq = move.to_square
        app.selected_sq = from_sq
        app.legal_targets = [m.to_square for m in app.board.legal_moves if m.from_square == from_sq]
        app._draw_board()
        capture_frame()

        # 2. Executed move frame (shows move applied, green highlight, move history log, material score)
        app.selected_sq = None
        app.legal_targets = []
        app._apply_move(move)
        app._draw_board()
        capture_frame()

    # Hold the final position for hold_end_sec (number of repeated frames at given fps)
    final_hold_frames = int(args.fps * args.hold_end_sec)
    print(f"[*] Game sequence complete. Holding final state for {final_hold_frames} frames ({args.hold_end_sec}s)...")

    for _ in range(final_hold_frames):
        capture_frame()

    root.destroy()

    # Compile frames to high-quality GIF with ffmpeg palette optimization
    print(f"[*] Encoding GIF with ffmpeg (fps={args.fps})...")
    palette_path = os.path.join(temp_dir, "palette.png")

    # Step 1: Generate optimal palette
    cmd_palette = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-vf", "palettegen=stats_mode=full",
        palette_path
    ]
    subprocess.run(cmd_palette, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 2: Render GIF using palette
    cmd_gif = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-i", palette_path,
        "-filter_complex", "paletteuse=dither=bayer:bayer_scale=3",
        args.output
    ]
    subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Cleanup temp directory
    shutil.rmtree(temp_dir)

    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[✓] GIF recording complete! Saved to {args.output} ({out_size_mb:.2f} MB)")


if __name__ == "__main__":
    args = parse_args()
    record_game_gif(args)
