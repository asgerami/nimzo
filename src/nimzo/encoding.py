import numpy as np
import chess

# Planes & Move Encoding Constants
NUM_PLANES = 19
NUM_MOVE_TYPES = 73
NUM_ACTIONS = 64 * NUM_MOVE_TYPES  # 4672

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# 8 Sliding Directions (dr, df) in oriented frame (+r is forward/north, +f is east)
SLIDING_DIRECTIONS = [
    (+1, 0),   # 0: N
    (+1, +1),  # 1: NE
    (0, +1),   # 2: E
    (-1, +1),  # 3: SE
    (-1, 0),   # 4: S
    (-1, -1),  # 5: SW
    (0, -1),   # 6: W
    (+1, -1)   # 7: NW
]

# 8 Knight Directions (dr, df)
KNIGHT_DIRECTIONS = [
    (+2, +1),  # 0
    (+1, +2),  # 1
    (-1, +2),  # 2
    (-2, +1),  # 3
    (-2, -1),  # 4
    (-1, -2),  # 5
    (+1, -2),  # 6
    (+2, -1)   # 7
]

# Underpromotion directions (df) for forward step (+1 rank)
UNDERPROMO_DIRECTIONS = [-1, 0, +1]  # Left capture, Forward, Right capture
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]  # 0: N, 1: B, 2: R

def orient_square(sq: int, side_to_move: chess.Color) -> int:
    """Return square index oriented from side-to-move's perspective."""
    if side_to_move == chess.WHITE:
        return sq
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    return chess.square(file, 7 - rank)

def unorient_square(oriented_sq: int, side_to_move: chess.Color) -> int:
    """Map oriented square index back to unoriented board square."""
    return orient_square(oriented_sq, side_to_move)

def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board into a (19, 8, 8) float32 numpy array oriented from side-to-move's perspective.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    stm = board.turn
    
    # Planes 0-5: Own pieces, Planes 6-11: Opponent pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            oriented_sq = orient_square(sq, stm)
            r = chess.square_rank(oriented_sq)
            f = chess.square_file(oriented_sq)
            
            p_idx = PIECE_TYPES.index(piece.piece_type)
            plane_offset = 0 if piece.color == stm else 6
            planes[plane_offset + p_idx, r, f] = 1.0
            
    # Planes 12-15: Castling rights
    if stm == chess.WHITE:
        own_ks = float(board.has_kingside_castling_rights(chess.WHITE))
        own_qs = float(board.has_queenside_castling_rights(chess.WHITE))
        opp_ks = float(board.has_kingside_castling_rights(chess.BLACK))
        opp_qs = float(board.has_queenside_castling_rights(chess.BLACK))
    else:
        own_ks = float(board.has_kingside_castling_rights(chess.BLACK))
        own_qs = float(board.has_queenside_castling_rights(chess.BLACK))
        opp_ks = float(board.has_kingside_castling_rights(chess.WHITE))
        opp_qs = float(board.has_queenside_castling_rights(chess.WHITE))
        
    planes[12, :, :] = own_ks
    planes[13, :, :] = own_qs
    planes[14, :, :] = opp_ks
    planes[15, :, :] = opp_qs
    
    # Plane 16: En passant target square
    if board.ep_square is not None:
        oriented_ep = orient_square(board.ep_square, stm)
        r = chess.square_rank(oriented_ep)
        f = chess.square_file(oriented_ep)
        planes[16, r, f] = 1.0
        
    # Plane 17: Halfmove clock / 100.0 clamped to [0, 1]
    planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    
    # Plane 18: All ones (board edge reference)
    planes[18, :, :] = 1.0
    
    return planes

def move_to_action(move: chess.Move, board: chess.Board) -> int:
    """
    Encode a chess.Move on the given board to an action index in [0, 4671].
    Handles Chess960 castling conversions automatically.
    """
    stm = board.turn
    
    # Normalize Chess960 castling moves (e.g. e1h1 -> e1g1)
    if board.is_castling(move):
        if board.turn == chess.WHITE:
            from_sq = chess.E1
            to_sq = chess.G1 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else chess.C1
        else:
            from_sq = chess.E8
            to_sq = chess.G8 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else chess.C8
        from_sq_oriented = orient_square(from_sq, stm)
        to_sq_oriented = orient_square(to_sq, stm)
        promotion = None
    else:
        from_sq_oriented = orient_square(move.from_square, stm)
        to_sq_oriented = orient_square(move.to_square, stm)
        promotion = move.promotion

    from_r, from_f = chess.square_rank(from_sq_oriented), chess.square_file(from_sq_oriented)
    to_r, to_f = chess.square_rank(to_sq_oriented), chess.square_file(to_sq_oriented)
    
    dr = to_r - from_r
    df = to_f - from_f
    
    # Underpromotions (Knight, Bishop, Rook)
    if promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        dir_idx = UNDERPROMO_DIRECTIONS.index(df)
        piece_idx = UNDERPROMO_PIECES.index(promotion)
        move_type = 64 + dir_idx * 3 + piece_idx
        return from_sq_oriented * NUM_MOVE_TYPES + move_type
        
    # Knight moves
    if (dr, df) in KNIGHT_DIRECTIONS:
        k_idx = KNIGHT_DIRECTIONS.index((dr, df))
        move_type = 56 + k_idx
        return from_sq_oriented * NUM_MOVE_TYPES + move_type
        
    # Sliding moves (including Queen promotions and Castling)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_f = 0 if df == 0 else (1 if df > 0 else -1)
    
    direction = (step_r, step_f)
    if direction in SLIDING_DIRECTIONS:
        dir_idx = SLIDING_DIRECTIONS.index(direction)
        distance = max(abs(dr), abs(df))
        move_type = dir_idx * 7 + (distance - 1)
        return from_sq_oriented * NUM_MOVE_TYPES + move_type
        
    raise ValueError(f"Unencodable move {move} (dr={dr}, df={df}, promo={promotion})")

def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """
    Decode an action index in [0, 4671] back into a chess.Move for the given board.
    """
    stm = board.turn
    from_sq_oriented = action // NUM_MOVE_TYPES
    move_type = action % NUM_MOVE_TYPES
    
    from_sq = unorient_square(from_sq_oriented, stm)
    from_r = chess.square_rank(from_sq_oriented)
    from_f = chess.square_file(from_sq_oriented)
    
    promotion = None
    
    if move_type < 56:
        # Sliding move
        dir_idx = move_type // 7
        distance = (move_type % 7) + 1
        dr_unit, df_unit = SLIDING_DIRECTIONS[dir_idx]
        dr = dr_unit * distance
        df = df_unit * distance
    elif move_type < 64:
        # Knight move
        k_idx = move_type - 56
        dr, df = KNIGHT_DIRECTIONS[k_idx]
    else:
        # Underpromotion
        u_idx = move_type - 64
        dir_idx = u_idx // 3
        piece_idx = u_idx % 3
        dr = 1
        df = UNDERPROMO_DIRECTIONS[dir_idx]
        promotion = UNDERPROMO_PIECES[piece_idx]
        
    to_r = from_r + dr
    to_f = from_f + df
    
    to_sq_oriented = chess.square(to_f, to_r)
    to_sq = unorient_square(to_sq_oriented, stm)
    
    # Detect Queen promotion (pawn moving to last rank without underpromotion flag)
    piece = board.piece_at(from_sq)
    if piece is not None and piece.piece_type == chess.PAWN:
        if (stm == chess.WHITE and chess.square_rank(to_sq) == 7) or (stm == chess.BLACK and chess.square_rank(to_sq) == 0):
            if promotion is None:
                promotion = chess.QUEEN
                
    move = chess.Move(from_sq, to_sq, promotion=promotion)
    
    # Handle castling mapping if board has castling
    if piece is not None and piece.piece_type == chess.KING:
        if abs(chess.square_file(to_sq) - chess.square_file(from_sq)) == 2:
            for legal in board.legal_moves:
                if board.is_castling(legal) and legal.from_square == from_sq:
                    if (legal.to_square > from_sq and to_sq > from_sq) or (legal.to_square < from_sq and to_sq < from_sq):
                        return legal
                        
    return move

def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Returns a boolean array of shape (4672,) where True indicates a legal move."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        action = move_to_action(move, board)
        mask[action] = True
    return mask
