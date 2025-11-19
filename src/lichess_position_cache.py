"""Lichess position streaming for dynamic DPO training.

This module streams positions from Lichess database for ephemeral training.

Data format:
  - Read from compressed zstd file (lichess_db_eval.jsonl.zst)
  - Each position contains: FEN-4 string, Stockfish evaluations with multiple Principal Variations (PVs)

Extraction:
  - FEN-4 → FEN-6 (append 0 20 for halfmove/fullmove counters)
  - preferred_move: first move of first PV (Stockfish's top choice)
  - good_moves: first move of ALL PVs (acceptable alternatives)

During training, we dynamically generate pairs by finding intruders:
  - Intruder = move with Q > Q(preferred) AND NOT in good_moves

Each position is used once then discarded (ephemeral, never cached).
"""

import io
import json
import os
from typing import Iterator, Optional

import chess
import numpy as np
import zstandard as zstd

from searchless_chess.src import tokenizer
from searchless_chess.src import utils


def stream_lichess_positions(database_path: str, max_positions: Optional[int] = None) -> Iterator[dict]:
  """Stream positions from compressed Lichess database.

  Args:
    database_path: Path to lichess_db_eval.jsonl.zst file.
    max_positions: Maximum number of positions to read (None = unlimited).

  Yields:
    Dictionary with 'fen' and 'evals' keys.
  """
  if not os.path.exists(database_path):
    raise FileNotFoundError(
        f"Lichess database not found at {database_path}. "
        "Download from https://database.lichess.org/lichess_db_eval.jsonl.zst"
    )

  dctx = zstd.ZstdDecompressor()

  with open(database_path, 'rb') as compressed:
    with dctx.stream_reader(compressed) as reader:
      text_stream = io.TextIOWrapper(reader, encoding='utf-8')
      for i, line in enumerate(text_stream):
        if max_positions and i >= max_positions:
          break
        yield json.loads(line)


def fen4_to_fen6(fen4: str) -> str:
  """Convert FEN-4 to FEN-6 with consistent halfmove/fullmove counters."""
  halfmove = 0
  fullmove = 20
  return f"{fen4} {halfmove} {fullmove}"


def extract_moves_from_evals(evals: list) -> tuple[str, list[str]]:
  """Extract preferred move and all good moves from Lichess evals.

  Args:
    evals: List of evaluation dictionaries from Lichess DB.

  Returns:
    Tuple of (preferred_move_uci, list_of_all_good_moves_uci).
  """
  # Use the evaluation with the most PVs (deepest analysis)
  best_eval = evals[-1]

  # Extract all moves from all PVs
  all_good_moves = []
  for pv in best_eval['pvs']:
    line = pv['line']
    first_move = line.split()[0]
    all_good_moves.append(first_move)

  # First PV is the preferred move
  preferred_move = all_good_moves[0]

  return preferred_move, all_good_moves


def generate_position_cache(
    database_path: str,
    cache_path: str,
    max_positions: Optional[int] = None,
    skip_errors: bool = True,
):
  """Generate position cache from Lichess database.

  This does NOT run the model - it only extracts positions and Stockfish moves.

  Args:
    database_path: Path to lichess_db_eval.jsonl.zst.
    cache_path: Path to save cache JSONL file.
    max_positions: Maximum positions to process.
    skip_errors: Whether to skip positions with errors.
  """
  cached_count = 0
  skipped_count = 0

  with open(cache_path, 'w') as cache_file:
    for position_data in stream_lichess_positions(database_path, max_positions):
      try:
        # Parse FEN
        fen4 = position_data['fen']
        fen6 = fen4_to_fen6(fen4)

        # Validate position
        board = chess.Board(fen6)
        if board.is_game_over():
          skipped_count += 1
          continue

        # Extract moves
        preferred_move, good_moves = extract_moves_from_evals(position_data['evals'])

        # Validate moves
        preferred_move_obj = chess.Move.from_uci(preferred_move)
        if preferred_move_obj not in board.legal_moves:
          skipped_count += 1
          continue

        # Tokenize position
        tokenized_fen = tokenizer.tokenize(fen6).astype(np.int32)

        # Convert moves to action indices
        preferred_action = utils.MOVE_TO_ACTION[preferred_move]
        good_actions = [utils.MOVE_TO_ACTION[move] for move in good_moves]

        # Write to cache
        cache_entry = {
            'position': tokenized_fen.tolist(),
            'preferred_move': preferred_action,
            'good_moves': good_actions,
        }
        cache_file.write(json.dumps(cache_entry) + '\n')
        cached_count += 1

        if cached_count % 10000 == 0:
          print(f"Cached {cached_count:,} positions (skipped {skipped_count:,})...")

      except Exception as e:
        if skip_errors:
          skipped_count += 1
          if skipped_count % 1000 == 0:
            print(f"Skipped {skipped_count:,} positions due to errors")
        else:
          raise

  print(f"\nCache generation complete!")
  print(f"  Cached: {cached_count:,} positions")
  print(f"  Skipped: {skipped_count:,} positions")
  print(f"  Saved to: {cache_path}")


def load_position_cache(cache_path: str, max_positions: Optional[int] = None) -> Iterator[dict]:
  """Load positions from cache file.

  Args:
    cache_path: Path to cache JSONL file.
    max_positions: Maximum positions to load.

  Yields:
    Dictionary with 'position', 'preferred_move', 'good_moves'.
  """
  with open(cache_path, 'r') as f:
    for i, line in enumerate(f):
      if max_positions and i >= max_positions:
        break
      yield json.loads(line)
