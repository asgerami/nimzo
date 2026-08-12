import time
import threading
import tkinter as tk
import customtkinter as ctk
import chess
from typing import Optional
from nimzo.model import build_model_from_config
from nimzo.engine import NimzoEngine

# Catppuccin Mocha -- one palette for the whole app, chrome and board alike.
COL_BASE = "#1E1E2E"
COL_MANTLE = "#181825"
COL_CRUST = "#11111B"
COL_SURFACE0 = "#313244"
COL_SURFACE1 = "#45475A"
COL_SURFACE2 = "#585B70"
COL_OVERLAY0 = "#6C7086"
COL_TEXT = "#CDD6F4"
COL_SUBTEXT = "#A6ADC8"
COL_BLUE = "#89B4FA"
COL_LAVENDER = "#B4BEFE"
COL_GREEN = "#A6E3A1"
COL_RED = "#F38BA8"
COL_PEACH = "#FAB387"
COL_YELLOW = "#F9E2AF"
COL_MAUVE = "#CBA6F7"

# Board squares live in the same family as the chrome, not a separate "wood" theme.
BOARD_LIGHT = "#DCE0F5"
BOARD_DARK = "#51576D"
BOARD_SELECTED = COL_YELLOW
BOARD_LAST_MOVE = "#93C9A6"
BOARD_LEGAL_DOT = COL_BLUE
BOARD_CAPTURE_RING = COL_RED

PIECE_SYMBOLS = {
    (chess.PAWN, chess.WHITE): "♟",
    (chess.KNIGHT, chess.WHITE): "♞",
    (chess.BISHOP, chess.WHITE): "♝",
    (chess.ROOK, chess.WHITE): "♜",
    (chess.QUEEN, chess.WHITE): "♛",
    (chess.KING, chess.WHITE): "♚",
    (chess.PAWN, chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK, chess.BLACK): "♜",
    (chess.QUEEN, chess.BLACK): "♛",
    (chess.KING, chess.BLACK): "♚",
}

STARTING_COUNTS = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
FILES = "abcdefgh"

class NimzoGUI:
    def __init__(
        self,
        root: ctk.CTk,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        canvas_size: int = 560,
    ):
        self.root = root
        self.root.title("Nimzo — Interactive Chess AI")
        self.canvas_size = canvas_size
        self.square_size = self.canvas_size // 8
        self.root.geometry(f"{self.canvas_size + 380}x{self.canvas_size + 160}")
        self.root.minsize(880, 640)
        self.root.configure(fg_color=COL_BASE)

        if not config_path or not weights_path:
            raise ValueError(
                "Nimzo GUI requires --config and --weights; there is no random-move fallback."
            )
        self.model = build_model_from_config(config_path)
        import torch
        ckpt = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Always run at max strength -- no difficulty setting.
        self.engine = NimzoEngine(model=self.model, use_mcts=True, playouts=1600)
        self.board = chess.Board()

        self.selected_sq: Optional[int] = None
        self.legal_targets: list[int] = []
        self.last_move: Optional[chess.Move] = None
        self.user_color = chess.WHITE
        self.is_engine_thinking = False

        self._setup_ui()
        self._draw_board()
        self._update_captured()

    def _setup_ui(self):
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- Left: board card ---
        board_card = ctk.CTkFrame(main_container, fg_color=COL_MANTLE, corner_radius=16)
        board_card.pack(side=tk.LEFT, padx=(0, 20), fill=tk.Y)



        self.canvas = tk.Canvas(
            board_card,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=COL_MANTLE,
            highlightthickness=0
        )
        self.canvas.pack(padx=16, pady=16)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Captured-pieces strip beneath the board
        captured_frame = ctk.CTkFrame(board_card, fg_color="transparent")
        captured_frame.pack(fill=tk.X, padx=16, pady=(0, 16))

        self.white_captured_label = ctk.CTkLabel(
            captured_frame, text="", font=("Dejavu Sans", 16), text_color=COL_SUBTEXT, anchor="w"
        )
        self.white_captured_label.pack(side=tk.LEFT, anchor="w")

        self.material_diff_label = ctk.CTkLabel(
            captured_frame, text="", font=("Helvetica", 12, "bold"), text_color=COL_GREEN
        )
        self.material_diff_label.pack(side=tk.RIGHT, anchor="e")

        # --- Right: sidebar ---
        panel = ctk.CTkFrame(main_container, fg_color="transparent")
        panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title_label = ctk.CTkLabel(
            panel, text="Nimzo", font=("Helvetica", 26, "bold"), text_color=COL_BLUE, anchor="w"
        )
        title_label.pack(anchor="w")
        subtitle_label = ctk.CTkLabel(
            panel, text="AlphaZero-style engine, built from scratch",
            font=("Helvetica", 12), text_color=COL_SUBTEXT, anchor="w"
        )
        subtitle_label.pack(anchor="w", pady=(0, 16))

        # Status card: colored dot + text
        status_card = ctk.CTkFrame(panel, fg_color=COL_SURFACE0, corner_radius=10)
        status_card.pack(fill=tk.X, pady=(0, 16))

        self.status_dot = ctk.CTkLabel(
            status_card, text="●", font=("Helvetica", 14), text_color=COL_GREEN, width=20
        )
        self.status_dot.pack(side=tk.LEFT, padx=(14, 0), pady=10)
        self.status_label = ctk.CTkLabel(
            status_card, text="Your turn (White)", font=("Helvetica", 13, "bold"),
            text_color=COL_TEXT, anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=(6, 14), pady=10, fill=tk.X, expand=True)

        # Game controls
        btn_frame = ctk.CTkFrame(panel, fg_color="transparent")
        btn_frame.pack(fill=tk.X, pady=(0, 16))

        self.new_game_btn = ctk.CTkButton(
            btn_frame, text="↻  New Game", command=self.new_game,
            fg_color=COL_BLUE, hover_color=COL_LAVENDER, text_color=COL_CRUST,
            font=("Helvetica", 12, "bold"), corner_radius=8, height=36
        )
        self.new_game_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.undo_btn = ctk.CTkButton(
            btn_frame, text="↶  Undo", command=self.undo_move,
            fg_color=COL_SURFACE1, hover_color=COL_RED, text_color=COL_TEXT,
            font=("Helvetica", 12, "bold"), corner_radius=8, height=36
        )
        self.undo_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

        # Move history
        ctk.CTkLabel(
            panel, text="MOVE HISTORY", font=("Helvetica", 11, "bold"),
            text_color=COL_OVERLAY0, anchor="w"
        ).pack(anchor="w", pady=(4, 6))

        self.move_log = ctk.CTkTextbox(
            panel, fg_color=COL_MANTLE, text_color=COL_TEXT,
            font=("Consolas", 12), corner_radius=10, wrap="none", state="disabled"
        )
        self.move_log.pack(fill=tk.BOTH, expand=True)

    def _draw_board(self):
        self.canvas.delete("all")

        for r in range(8):
            for f in range(8):
                sq = chess.square(f, 7 - r) if self.user_color == chess.WHITE else chess.square(7 - f, r)
                x1 = f * self.square_size
                y1 = r * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                is_dark = (r + f) % 2 == 1
                color = BOARD_DARK if is_dark else BOARD_LIGHT

                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    color = BOARD_LAST_MOVE
                if sq == self.selected_sq:
                    color = BOARD_SELECTED

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                # Coordinate labels inset in the edge squares (lichess/chess.com style)
                label_color = BOARD_DARK if not is_dark else BOARD_LIGHT
                if f == 0:
                    rank_label = str(8 - r) if self.user_color == chess.WHITE else str(r + 1)
                    self.canvas.create_text(
                        x1 + 8, y1 + 9, text=rank_label, font=("Helvetica", 11, "bold"),
                        fill=label_color, anchor="nw"
                    )
                if r == 7:
                    file_label = FILES[f] if self.user_color == chess.WHITE else FILES[7 - f]
                    self.canvas.create_text(
                        x2 - 8, y2 - 7, text=file_label, font=("Helvetica", 11, "bold"),
                        fill=label_color, anchor="se"
                    )

                # Legal move target indicator
                if sq in self.legal_targets:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if self.board.piece_at(sq) is None:
                        radius = 11
                        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=BOARD_LEGAL_DOT, outline="")
                    else:
                        radius = self.square_size / 2 - 3
                        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=BOARD_CAPTURE_RING, width=3)

                # Piece rendering. A real outline (not just a diagonal drop
                # shadow) so pieces stay readable regardless of square color --
                # pure white pieces on the near-white light squares had almost
                # no contrast without this.
                piece = self.board.piece_at(sq)
                if piece:
                    symbol = PIECE_SYMBOLS.get((piece.piece_type, piece.color), "")
                    fg_color = "#FFFFFF" if piece.color == chess.WHITE else COL_CRUST
                    outline_color = COL_CRUST if piece.color == chess.WHITE else "#F5F5F5"

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    font_size = int(self.square_size * 0.6)
                    font = ("Dejavu Sans", font_size, "bold")

                    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
                        self.canvas.create_text(cx + dx, cy + dy, text=symbol, font=font, fill=outline_color)
                    self.canvas.create_text(cx, cy, text=symbol, font=font, fill=fg_color)

    def _update_captured(self):
        white_captured = []
        black_captured = []
        for piece_type, start_count in STARTING_COUNTS.items():
            black_remaining = len(self.board.pieces(piece_type, chess.BLACK))
            white_remaining = len(self.board.pieces(piece_type, chess.WHITE))
            white_captured.extend([piece_type] * (start_count - black_remaining))
            black_captured.extend([piece_type] * (start_count - white_remaining))

        white_captured.sort(key=lambda p: -PIECE_VALUES[p])
        black_captured.sort(key=lambda p: -PIECE_VALUES[p])

        white_text = "".join(PIECE_SYMBOLS[(p, chess.BLACK)] for p in white_captured)
        black_text = "".join(PIECE_SYMBOLS[(p, chess.WHITE)] for p in black_captured)
        self.white_captured_label.configure(text=f"{white_text}   {black_text}")

        diff = sum(PIECE_VALUES[p] for p in white_captured) - sum(PIECE_VALUES[p] for p in black_captured)
        if diff > 0:
            self.material_diff_label.configure(text=f"White +{diff}", text_color=COL_GREEN)
        elif diff < 0:
            self.material_diff_label.configure(text=f"Black +{-diff}", text_color=COL_RED)
        else:
            self.material_diff_label.configure(text="Even", text_color=COL_SUBTEXT)

    def _on_canvas_click(self, event):
        if self.is_engine_thinking or self.board.is_game_over():
            return

        if self.board.turn != self.user_color:
            return

        f = int(event.x // self.square_size)
        r = int(event.y // self.square_size)
        sq = chess.square(f, 7 - r) if self.user_color == chess.WHITE else chess.square(7 - f, r)

        piece = self.board.piece_at(sq)

        if self.selected_sq is None:
            if piece and piece.color == self.user_color:
                self.selected_sq = sq
                self.legal_targets = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
        else:
            if sq == self.selected_sq:
                self.selected_sq = None
                self.legal_targets = []
            elif sq in self.legal_targets:
                # Execute human move
                move = self._get_move_with_promotion(self.selected_sq, sq)
                if move and move in self.board.legal_moves:
                    self._apply_move(move)
                    self.selected_sq = None
                    self.legal_targets = []

                    if not self.board.is_game_over():
                        self._trigger_engine_turn()
            elif piece and piece.color == self.user_color:
                self.selected_sq = sq
                self.legal_targets = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
            else:
                self.selected_sq = None
                self.legal_targets = []

        self._draw_board()

    def _get_move_with_promotion(self, from_sq: int, to_sq: int) -> Optional[chess.Move]:
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0):
                promo_dialog = ctk.CTkToplevel(self.root)
                promo_dialog.title("Promote Pawn")
                promo_dialog.geometry("340x120")
                promo_dialog.configure(fg_color=COL_BASE)
                promo_dialog.transient(self.root)
                promo_dialog.resizable(False, False)

                selected_promo = [chess.QUEEN]

                def choose(p):
                    selected_promo[0] = p
                    promo_dialog.destroy()

                ctk.CTkLabel(
                    promo_dialog, text="Promote to:", font=("Helvetica", 12, "bold"), text_color=COL_TEXT
                ).pack(pady=(14, 8))
                f_btns = ctk.CTkFrame(promo_dialog, fg_color="transparent")
                f_btns.pack(pady=4)

                choices = [
                    ("♛", chess.QUEEN), ("♜", chess.ROOK),
                    ("♝", chess.BISHOP), ("♞", chess.KNIGHT),
                ]
                for symbol, piece_type in choices:
                    ctk.CTkButton(
                        f_btns, text=symbol, width=48, height=48, font=("Dejavu Sans", 20, "bold"),
                        fg_color=COL_SURFACE0, hover_color=COL_BLUE, text_color=COL_TEXT,
                        corner_radius=8, command=lambda p=piece_type: choose(p)
                    ).pack(side=tk.LEFT, padx=4)

                promo_dialog.update_idletasks()
                promo_dialog.deiconify()
                self.root.wait_window(promo_dialog)
                return chess.Move(from_sq, to_sq, promotion=selected_promo[0])

        return chess.Move(from_sq, to_sq)

    def _apply_move(self, move: chess.Move):
        move_color = self.board.turn
        san = self.board.san(move)
        move_num = (len(self.board.move_stack) // 2) + 1

        self.board.push(move)
        self.last_move = move

        self.move_log.configure(state="normal")
        if move_color == chess.WHITE:
            self.move_log.insert("end", f"{move_num}. {san}")
        else:
            self.move_log.insert("end", f"  {san}\n")
        self.move_log.see("end")
        self.move_log.configure(state="disabled")

        self._update_captured()

        if self.board.is_game_over():
            self._update_status(f"Game Over! Result: {self.board.result()}", COL_RED)
        else:
            turn_str = "White" if self.board.turn == chess.WHITE else "Black"
            self._update_status(f"Turn: {turn_str}", COL_GREEN)

    def _trigger_engine_turn(self):
        self.is_engine_thinking = True
        self.new_game_btn.configure(state=tk.DISABLED)
        self.undo_btn.configure(state=tk.DISABLED)
        self._update_status("Nimzo is thinking...", COL_PEACH)

        threading.Thread(target=self._engine_worker, daemon=True).start()

    def _engine_worker(self):
        t0 = time.time()
        mcts_move = self.engine.get_best_move(self.board, movetime_ms=1000)

        # Ensure minimum 300ms delay so UI transition feels natural
        elapsed = time.time() - t0
        if elapsed < 0.3:
            time.sleep(0.3 - elapsed)

        self.root.after(0, self._on_engine_move_computed, mcts_move)

    def _on_engine_move_computed(self, move: chess.Move):
        self.is_engine_thinking = False
        self.new_game_btn.configure(state=tk.NORMAL)
        self.undo_btn.configure(state=tk.NORMAL)

        self._apply_move(move)
        self._draw_board()

    def _update_status(self, text: str, color: str = COL_GREEN):
        self.status_label.configure(text=text)
        self.status_dot.configure(text_color=color)

    def _rewrite_move_log(self):
        self.move_log.configure(state="normal")
        self.move_log.delete("1.0", "end")
        moves = self.board.move_stack
        replay = chess.Board()
        line = ""
        for i, move in enumerate(moves):
            san = replay.san(move)
            replay.push(move)
            if i % 2 == 0:
                line = f"{i // 2 + 1}. {san}"
            else:
                self.move_log.insert("end", f"{line}  {san}\n")
                line = ""
        if line:
            self.move_log.insert("end", line)
        self.move_log.see("end")
        self.move_log.configure(state="disabled")

    def undo_move(self):
        if self.is_engine_thinking:
            return

        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            self.last_move = self.board.peek() if self.board.move_stack else None
            self._update_status("Undid last turn.", COL_GREEN)
        elif len(self.board.move_stack) == 1:
            self.board.pop()
            self.last_move = None
            self._update_status("Undid move.", COL_GREEN)

        self._rewrite_move_log()
        self._update_captured()
        self.selected_sq = None
        self.legal_targets = []
        self._draw_board()

    def new_game(self):
        if self.is_engine_thinking:
            return

        self.board = chess.Board()
        self.selected_sq = None
        self.legal_targets = []
        self.last_move = None
        self.move_log.configure(state="normal")
        self.move_log.delete("1.0", "end")
        self.move_log.configure(state="disabled")
        self._update_status("New Game started. Your turn (White).", COL_GREEN)
        self._update_captured()
        self._draw_board()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nimzo GUI Chess App")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    args = parser.parse_args()

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = NimzoGUI(root, config_path=args.config, weights_path=args.weights)
    root.mainloop()

if __name__ == "__main__":
    main()
