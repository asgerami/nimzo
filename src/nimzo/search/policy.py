import torch
import numpy as np
import chess
from nimzo.model import ChessResNet
from nimzo.encoding import board_to_planes, move_to_action, action_to_move, get_legal_move_mask

def best_move_policy(model: ChessResNet, board: chess.Board, device: str = "cpu") -> chess.Move:
    """Fast raw policy prediction (< 5ms on CPU)."""
    model.eval()
    planes = board_to_planes(board)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        p_logits, _ = model(x)
        logits = p_logits.squeeze(0).cpu().numpy()
        
    mask = get_legal_move_mask(board)
    logits[~mask] = -1e9
    
    best_action = int(np.argmax(logits))
    return action_to_move(best_action, board)
