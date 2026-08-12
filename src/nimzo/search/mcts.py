import math
import time
import torch
import numpy as np
import chess
from collections import OrderedDict
from typing import Dict, Hashable, List, Optional, Tuple
from nimzo.model import ChessResNet
from nimzo.encoding import board_to_planes, move_to_action, action_to_move, get_legal_move_mask
from nimzo.baselines import evaluate_psqt, PIECE_VALUES

class MCTSNode:
    def __init__(self, prior_p: float = 0.0):
        self.prior_p = prior_p
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    def __init__(self, model: Optional[ChessResNet] = None, c_puct: float = 1.5, batch_size: int = 16, tt_max_size: int = 200_000):
        self.model = model
        self.c_puct = c_puct
        self.batch_size = batch_size
        # LRU transposition table. Keyed on board._transposition_key(), which
        # excludes halfmove_clock -- an accepted standard TT limitation, not
        # worth working around (would weaken merging for no real gain).
        self.transposition_table: "OrderedDict[Hashable, MCTSNode]" = OrderedDict()
        self.tt_max_size = tt_max_size
        self.root: Optional[MCTSNode] = None

    def _tt_get(self, key: Hashable) -> Optional[MCTSNode]:
        node = self.transposition_table.get(key)
        if node is not None:
            self.transposition_table.move_to_end(key)
        return node

    def _tt_put(self, key: Hashable, node: MCTSNode) -> None:
        self.transposition_table[key] = node
        self.transposition_table.move_to_end(key)
        while len(self.transposition_table) > self.tt_max_size:
            self.transposition_table.popitem(last=False)

    def advance_root(self, board_before: chess.Board, move: chess.Move) -> None:
        """Advance the tree to the child reached by `move`, discarding the rest
        of the tree, so a game's search statistics compound across moves
        instead of restarting from scratch every turn. Falls back to a fresh
        tree (self.root = None) if there is no existing root, or `move` was
        never expanded as a child (e.g. pruned, zero visits)."""
        if self.root is None:
            return
        action = move_to_action(move, board_before)
        self.root = self.root.children.get(action)

    def search(self, board: chess.Board, playouts: int = 800, time_limit_sec: float = 1.0) -> chess.Move:
        """Run PUCT MCTS search and return the move with the highest visit count."""
        if self.root is None:
            root_key = board._transposition_key()
            existing = self._tt_get(root_key)
            if existing is not None:
                self.root = existing
            else:
                self.root = MCTSNode()
                self._tt_put(root_key, self.root)

        t0 = time.time()
        playout_count = 0
        
        while playout_count < playouts:
            if time_limit_sec > 0 and (time.time() - t0) >= time_limit_sec:
                break
                
            # Collect a batch of leaves
            batch_paths = []
            for _ in range(min(self.batch_size, playouts - playout_count)):
                node = self.root
                curr_board = board.copy()
                path = [node]
                
                # Selection loop
                while node.is_expanded and not curr_board.is_game_over(claim_draw=True):
                    best_action, best_child = self._select_child(node, curr_board)
                    if best_child is None:
                        break
                    curr_board.push(action_to_move(best_action, curr_board))

                    # Transposition merge: if this position was already reached
                    # via a different move order elsewhere in the tree, redirect
                    # this edge to the existing node instead of duplicating it.
                    child_key = curr_board._transposition_key()
                    existing = self._tt_get(child_key)
                    if existing is not None and existing is not best_child:
                        node.children[best_action] = existing
                        best_child = existing
                    else:
                        self._tt_put(child_key, best_child)

                    # Apply virtual loss to child
                    best_child.visit_count += 1
                    best_child.value_sum -= 1.0
                    node = best_child
                    path.append(node)
                    
                batch_paths.append((path, curr_board))
                playout_count += 1
                
            # Evaluate batch of leaves
            self._evaluate_and_backprop_batch(batch_paths)

        # Select action with highest visit count
        best_action = None
        best_visits = -1
        for action, child in self.root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
                
        if best_action is None:
            return list(board.legal_moves)[0]
        return action_to_move(best_action, board)

    def _select_child(self, node: MCTSNode, board: chess.Board) -> Tuple[int, Optional[MCTSNode]]:
        best_score = -1e9
        best_action = None
        best_child = None
        
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(max(1, total_visits))
        
        for action, child in node.children.items():
            u = self.c_puct * child.prior_p * sqrt_total / (1 + child.visit_count)
            # Negamax selection: -child.q_value gives Q score from parent's perspective
            score = -child.q_value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def _evaluate_and_backprop_batch(self, batch_paths: List[Tuple[List[MCTSNode], chess.Board]]):
        planes_list = []
        evaluate_indices = []
        
        for idx, (path, board) in enumerate(batch_paths):
            leaf = path[-1]
            
            # Check terminal state (claim_draw=True: also catches fifty-move-rule
            # and threefold-repetition draws, which the network should never see)
            if board.is_game_over(claim_draw=True):
                leaf.is_terminal = True
                res = board.result(claim_draw=True)
                if res == "1-0":
                    leaf.terminal_value = 1.0 if board.turn == chess.BLACK else -1.0
                elif res == "0-1":
                    leaf.terminal_value = -1.0 if board.turn == chess.BLACK else 1.0
                else:
                    leaf.terminal_value = 0.0
                val = leaf.terminal_value
                self._backprop(path, val)
            elif self.model is None:
                # Heuristic PSQT + Tactical Evaluation Fallback (when no trained ResNet loaded)
                cp_val = evaluate_psqt(board)
                # Centipawns from side-to-move's perspective
                stm_cp = cp_val if board.turn == chess.WHITE else -cp_val
                val = math.tanh(stm_cp / 400.0)
                
                # Compute tactical move priors (captures & checks get boosted priors)
                legal_moves = list(board.legal_moves)
                scores = []
                for move in legal_moves:
                    score = 1.0
                    if board.is_capture(move):
                        captured = board.piece_at(move.to_square)
                        cap_val = PIECE_VALUES.get(captured.piece_type, 100) if captured else 100
                        score += 10.0 + (cap_val / 50.0)
                    if board.gives_check(move):
                        score += 5.0
                    scores.append(score)
                    
                sum_scores = sum(scores)
                for move, s in zip(legal_moves, scores):
                    act = move_to_action(move, board)
                    child = MCTSNode(prior_p=s / sum_scores)
                    leaf.children[act] = child
                    
                leaf.is_expanded = True
                self._backprop(path, val)
            else:
                planes = board_to_planes(board)
                planes_list.append(planes)
                evaluate_indices.append(idx)

        if not planes_list or self.model is None:
            return

        # Neural network batch forward pass
        self.model.eval()
        x = torch.from_numpy(np.array(planes_list, dtype=np.float32))
        with torch.no_grad():
            p_logits, v_logits = self.model(x)
            p_probs = torch.softmax(p_logits, dim=-1).cpu().numpy()
            v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()

        for i, idx in enumerate(evaluate_indices):
            path, board = batch_paths[idx]
            leaf = path[-1]
            probs = p_probs[i]
            v_prob = v_probs[i]  # [win, draw, loss]
            value = v_prob[0] - v_prob[2]
            
            # Expand leaf
            mask = get_legal_move_mask(board)
            legal_probs = probs * mask
            sum_p = legal_probs.sum()
            if sum_p > 0:
                legal_probs /= sum_p
            else:
                legal_probs[mask] = 1.0 / mask.sum()

            for move in board.legal_moves:
                act = move_to_action(move, board)
                child = MCTSNode(prior_p=float(legal_probs[act]))
                leaf.children[act] = child
                
            leaf.is_expanded = True
            self._backprop(path, value)

    def _backprop(self, path: List[MCTSNode], value: float):
        for idx, node in enumerate(reversed(path)):
            if node != path[0]:  # Non-root child nodes had virtual loss applied during selection
                node.value_sum += 1.0  # Undo virtual loss (-1.0)
            else:
                node.visit_count += 1  # Root visit count increment
                
            node.value_sum += value
            value = -value  # Negate for parent perspective
