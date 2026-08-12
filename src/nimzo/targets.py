import numpy as np

def cp_to_wdl(cp: float, is_mate: bool = False, draw_band: float = 50.0) -> np.ndarray:
    """
    Convert centipawn evaluation (from side-to-move's perspective) into a 3-class WDL distribution [win, draw, loss].
    
    Parameters:
    - cp: Centipawns (positive = side-to-move advantage, negative = opponent advantage) or mate in N moves.
    - is_mate: Boolean flag indicating if cp represents forced mate (e.g. +3 = mate in 3 moves for stm).
    - draw_band: Centipawn range within which draw probability is concentrated.
    
    Returns:
    - np.ndarray of shape (3,) with [p_win, p_draw, p_loss] summing to 1.0.
    """
    if is_mate:
        if cp > 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif cp < 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
    # Win probability curve from Elo model
    win_prob = 1.0 / (1.0 + 10.0 ** (-cp / 400.0))
    loss_prob = 1.0 / (1.0 + 10.0 ** (cp / 400.0))
    
    # Draw probability peaks near 0 centipawns
    draw_factor = np.exp(-((cp / draw_band) ** 2)) * 0.5
    
    # Normalize probabilities
    raw_win = max(0.0, win_prob - 0.5 * draw_factor)
    raw_loss = max(0.0, loss_prob - 0.5 * draw_factor)
    raw_draw = draw_factor
    
    total = raw_win + raw_draw + raw_loss
    return np.array([raw_win / total, raw_draw / total, raw_loss / total], dtype=np.float32)
