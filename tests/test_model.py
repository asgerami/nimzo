import os
import pytest
import torch
import numpy as np
import chess
from nimzo.model import ChessResNet, build_model_from_config
from nimzo.data.shards import encode_record, ShardDataset, RECORD_SIZE
from nimzo.encoding import move_to_action

def test_model_forward_pass_shapes():
    model = ChessResNet(num_blocks=2, channels=32)
    x = torch.randn(4, 19, 8, 8)
    p_logits, v_logits = model(x)
    assert p_logits.shape == (4, 4672)
    assert v_logits.shape == (4, 3)

def test_tiny_model_overfit_1000_positions_gate(tmp_path):
    """Milestone M3 Gate: tiny config reaches >99% policy accuracy on 1,000 held-in positions."""
    shard_file = tmp_path / "overfit_1000.bin"
    
    # Generate 1,000 deterministic test records
    records = []
    board = chess.Board()
    for idx in range(1000):
        legal_moves = list(board.legal_moves)
        move = legal_moves[idx % len(legal_moves)]
        action = move_to_action(move, board)
        rec = encode_record(board, action, cp=100, source=0)
        records.append(rec)
        board.push(move)
        if board.is_game_over():
            board.reset()
            
    with open(shard_file, "wb") as f:
        for r in records:
            f.write(r)
            
    dataset = ShardDataset(str(shard_file))
    assert len(dataset) == 1000
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ChessResNet(num_blocks=4, channels=48)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    
    print("\n[M3 Gate Overfit Test] Training tiny model on 1,000 positions...")
    final_acc = 0.0
    for epoch in range(1, 151):
        total_correct = 0
        total_samples = 0
        for planes, actions, wdl_targets, _sources in loader:
            optimizer.zero_grad()
            p_logits, v_logits = model(planes)
            p_loss = torch.nn.functional.cross_entropy(p_logits, actions)
            v_loss = torch.nn.functional.cross_entropy(v_logits, wdl_targets)
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()
            
            preds = p_logits.argmax(dim=-1)
            total_correct += (preds == actions).sum().item()
            total_samples += len(actions)
            
        final_acc = total_correct / total_samples
        if epoch % 30 == 0 or final_acc >= 0.99:
            print(f"Epoch {epoch}/150 | Policy Accuracy: {final_acc*100:.2f}%")
        if final_acc >= 0.99:
            break
            
    print(f"\n[M3 Gate Result] Final Policy Accuracy: {final_acc*100:.2f}%")
    assert final_acc >= 0.99
