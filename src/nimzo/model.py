import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Tuple

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_blocks: int = 6, channels: int = 64, in_channels: int = 19, policy_channels: int = 73, value_channels: int = 8):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(channels)
        
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        
        # Policy Head: Conv 1x1 C -> 73 -> flatten to 4672 logits
        self.policy_conv = nn.Conv2d(channels, policy_channels, kernel_size=1)
        
        # Value Head: Conv 1x1 C -> 8 -> BN -> ReLU -> FC 512 -> 128 -> 3 WDL logits
        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Input: (B, 19, 8, 8)
        Returns:
            policy_logits: (B, 4672)
            value_logits: (B, 3)  [win, draw, loss]
        """
        out = F.relu(self.in_bn(self.in_conv(x)))
        for block in self.res_blocks:
            out = block(out)
            
        # Policy Head
        p_out = self.policy_conv(out)  # (B, 73, 8, 8)
        policy_logits = p_out.flatten(start_dim=1)  # (B, 4672)
        
        # Value Head
        v_out = F.relu(self.value_bn(self.value_conv(out)))  # (B, 8, 8, 8)
        v_flat = v_out.flatten(start_dim=1)  # (B, 512)
        v_hidden = F.relu(self.value_fc1(v_flat))
        value_logits = self.value_fc2(v_hidden)  # (B, 3)
        
        return policy_logits, value_logits

def build_model_from_config(config_path: str) -> ChessResNet:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = ChessResNet(
        num_blocks=cfg.get("num_blocks", 6),
        channels=cfg.get("channels", 64),
        in_channels=cfg.get("in_channels", 19),
        policy_channels=cfg.get("policy_channels", 73),
        value_channels=cfg.get("value_channels", 8)
    )
    return model
