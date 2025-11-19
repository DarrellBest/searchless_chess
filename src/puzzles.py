# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates engines on the puzzles dataset from lichess."""

from collections.abc import Sequence
import io
import os

from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import pandas as pd

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib


_NUM_PUZZLES = flags.DEFINE_integer(
    name='num_puzzles',
    default=100,
    help='The number of puzzles to evaluate.',
)
_AGENT = flags.DEFINE_enum(
    name='agent',
    default='9M',
    enum_values=[
        'local',
        '9M',
        '136M',
        '270M',
        '9M_selfplay',
        '136M_selfplay',
        '270M_selfplay',
        '9M_lichess_best',
        '9M_lichess_100k',
        '9M_lichess_200k',
        '9M_lichess_300k',
        '9M_lichess_400k',
        '9M_lichess_500k',
        '9M_lichess_600k',
        '9M_lichess_700k',
        '9M_lichess_800k',
        '9M_lichess_900k',
        '9M_lichess_1000k',
    ] + [f'9M_stream_{step}' for step in [
        '1002', '2039', '3102', '4259', '5360', '6380', '7398', '8423', '9487',
        '10529', '11646', '12775', '13899', '14945', '15985', '16995', '18153',
        '19225', '20367', '21381', '22423', '23543', '24547', '25606', '26669',
        '27695', '28789', '29842', '30899', '31944', '32991', '34031', '35162',
        '36254', '37305', '38354', '39498', '40675', '41786', '42847', '43911',
        '44954', '46058', '47215', '48378', '49513', '50643', '51653', '52701',
        '53718', '54744', '55872', '56890', '57991', '59019', '60027', '61076',
        '62081', '63160', '64250', '65281', '66422', '67614', '68764', '69824',
        '70906', '71968', '72970', '74014', '75149', '76218', '77300', '78361',
        '79463', '80625', '81908', '82942', '84119', '85175', '86253', '87297',
        '88298', '89396', '90411', '91526',
    ]] + [
        'stockfish',
        'stockfish_all_moves',
        'leela_chess_zero_depth_1',
        'leela_chess_zero_policy_net',
        'leela_chess_zero_400_sims',
    ],
    help='The agent to evaluate.',
)


def evaluate_puzzle_from_pandas_row(
    puzzle: pd.Series,
    engine: engine_lib.Engine,
) -> bool:
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
  if game is None:
    raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
  board = game.end().board()
  return evaluate_puzzle_from_board(
      board=board,
      moves=puzzle['Moves'].split(' '),
      engine=engine,
  )


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine: engine_lib.Engine,
) -> bool:
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  for move_idx, move in enumerate(moves):
    # According to https://database.lichess.org/#puzzles, the FEN is the
    # position before the opponent makes their move. The position to present to
    # the player is after applying the first move to that FEN. The second move
    # is the beginning of the solution.
    if move_idx % 2 == 1:
      predicted_move = engine.play(board=board).uci()
      # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
      # check if the `predicted_move` results in a checkmate if it differs from
      # the solution.
      if move != predicted_move:
        board.push(chess.Move.from_uci(predicted_move))
        return board.is_checkmate()
    board.push(chess.Move.from_uci(move))
  return True


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  puzzles_path = os.path.join(
      os.getcwd(),
      '../data/puzzles.csv',
  )
  puzzles = pd.read_csv(puzzles_path, nrows=_NUM_PUZZLES.value)
  engine = constants.ENGINE_BUILDERS[_AGENT.value]()

  for puzzle_id, puzzle in puzzles.iterrows():
    correct = evaluate_puzzle_from_pandas_row(
        puzzle=puzzle,
        engine=engine,
    )
    print(
        {'puzzle_id': puzzle_id, 'correct': correct, 'rating': puzzle['Rating']}
    )

  # Cleanup: close the engine if it has a close method (e.g., Stockfish)
  if hasattr(engine, 'close'):
    engine.close()


if __name__ == '__main__':
  app.run(main)
