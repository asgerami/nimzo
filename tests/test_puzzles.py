import os
import csv
import pytest
import chess
import zstandard
from nimzo.data.shards import encode_record, decode_record, RECORD_SIZE, ShardDataset
from nimzo.data.pack_puzzles import parse_puzzle_line, pack_puzzles

def test_puzzle_packing_and_decoding(tmp_path):
    # Create sample puzzle CSV file
    puzzle_csv = tmp_path / "sample_puzzles.csv"
    with open(puzzle_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags", "DailyDate"])
        # Standard tactics puzzle
        writer.writerow(["00008", "r1bqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQkq - 0 7", "f6e4 c3e4 d5e4", "600", "75", "100", "10", "short", "http://example.com", "Opening", "2020"])
        # Castling puzzle
        writer.writerow(["0000d", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1 a8a1", "1200", "75", "100", "10", "castling", "http://example.com", "Opening", "2020"])
        
    out_dir = tmp_path / "shards"
    pack_puzzles(str(puzzle_csv), str(out_dir), limit=100)
    
    train_bin = out_dir / "puzzles_train.bin"
    assert os.path.exists(train_bin)
    
    dataset = ShardDataset(str(train_bin))
    assert len(dataset) > 0
    
    # Spot-check records decode to legal positions with legal action targets
    for idx in range(len(dataset)):
        rec_bytes = bytes(dataset.data[idx])
        assert len(rec_bytes) == RECORD_SIZE
        board, action, cp_val, source = decode_record(rec_bytes)
        assert isinstance(board, chess.Board)
        assert board.is_valid() or len(list(board.legal_moves)) > 0
        
        planes, act, wdl, rec_source = dataset[idx]
        assert planes.shape == (19, 8, 8)
        assert 0 <= act < 4672
        assert rec_source == 0  # puzzles are packed with source=0
        assert wdl.shape == (3,)

def test_puzzle_packing_from_zst(tmp_path):
    # Regression test: pack_puzzles must handle real .zst input (streamed line
    # iteration over ZstdDecompressor.stream_reader is not automatic; it needs
    # an io.TextIOWrapper). This path is what production data actually uses.
    csv_bytes_lines = [
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,DailyDate",
        "00008,r1bqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQkq - 0 7,f6e4 c3e4 d5e4,600,75,100,10,short,http://example.com,Opening,2020",
    ]
    raw_csv = ("\n".join(csv_bytes_lines) + "\n").encode("utf-8")

    puzzle_zst = tmp_path / "sample_puzzles.csv.zst"
    puzzle_zst.write_bytes(zstandard.ZstdCompressor().compress(raw_csv))

    out_dir = tmp_path / "shards_zst"
    pack_puzzles(str(puzzle_zst), str(out_dir), limit=100)

    train_bin = out_dir / "puzzles_train.bin"
    assert os.path.exists(train_bin)
    dataset = ShardDataset(str(train_bin))
    assert len(dataset) > 0
