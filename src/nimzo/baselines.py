import random
import chess

# Standard Piece Values (Centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Simplified Piece-Square Tables (oriented for White; flip rank for Black)
PAWN_PSQT = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_PSQT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]

def random_legal(board: chess.Board) -> chess.Move:
    """Select a uniform random legal move."""
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available")
    return random.choice(legal)

def evaluate_material(board: chess.Board) -> int:
    """Compute material balance from White's perspective."""
    val = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            v = PIECE_VALUES[piece.piece_type]
            val += v if piece.color == chess.WHITE else -v
    return val

def greedy_material(board: chess.Board) -> chess.Move:
    """1-ply greedy move selection maximizing material balance."""
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available")
        
    best_move = legal[0]
    best_eval = -999999 if board.turn == chess.WHITE else 999999
    
    for move in legal:
        board.push(move)
        score = evaluate_material(board)
        board.pop()
        
        if board.turn == chess.WHITE:
            if score > best_eval:
                best_eval = score
                best_move = move
        else:
            if score < best_eval:
                best_eval = score
                best_move = move
                
    return best_move

def evaluate_psqt(board: chess.Board) -> int:
    """Evaluate position using Material + Piece-Square Tables (White perspective)."""
    if board.is_checkmate():
        return -30000 if board.turn == chess.WHITE else 30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            v = PIECE_VALUES[piece.piece_type]
            
            # Add PSQT bonus for pawns and knights
            r = chess.square_rank(sq) if piece.color == chess.WHITE else (7 - chess.square_rank(sq))
            f = chess.square_file(sq)
            psqt_sq = r * 8 + f
            
            psqt_bonus = 0
            if piece.piece_type == chess.PAWN:
                psqt_bonus = PAWN_PSQT[psqt_sq]
            elif piece.piece_type == chess.KNIGHT:
                psqt_bonus = KNIGHT_PSQT[psqt_sq]
                
            total_v = v + psqt_bonus
            score += total_v if piece.color == chess.WHITE else -total_v
            
    return score

def minimax_alphabeta(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate_psqt(board)
        
    if board.turn == chess.WHITE:
        max_eval = -999999
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax_alphabeta(board, depth - 1, alpha, beta)
            board.pop()
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 999999
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax_alphabeta(board, depth - 1, alpha, beta)
            board.pop()
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval

def minimax_psqt(board: chess.Board, depth: int = 3) -> chess.Move:
    """3-ply alpha-beta minimax search using PSQT + material evaluation."""
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available")
        
    best_move = legal[0]
    alpha = -999999
    beta = 999999
    
    if board.turn == chess.WHITE:
        best_val = -999999
        for move in legal:
            board.push(move)
            val = minimax_alphabeta(board, depth - 1, alpha, beta)
            board.pop()
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
    else:
        best_val = 999999
        for move in legal:
            board.push(move)
            val = minimax_alphabeta(board, depth - 1, alpha, beta)
            board.pop()
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, best_val)
            
    return best_move
