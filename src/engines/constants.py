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

"""Constants for the engines."""

import functools
import os

import chess
import chess.engine
import chess.pgn
from jax import random as jrandom
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import lc0_engine
from searchless_chess.src.engines import neural_engines
from searchless_chess.src.engines import stockfish_engine


def _build_neural_engine(
    model_name: str,
    checkpoint_step: int = -1,
) -> neural_engines.NeuralEngine:
  """Returns a neural engine."""

  match model_name:
    case '9M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 256
      num_heads = 8
    case '136M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 1024
      num_heads = 8
    case '270M':
      policy = 'action_value'
      num_layers = 16
      embedding_dim = 1024
      num_heads = 8
    case 'local':
      policy = 'action_value'
      num_layers = 4
      embedding_dim = 64
      num_heads = 4
    case _:
      raise ValueError(f'Unknown model: {model_name}')

  num_return_buckets = 128

  match policy:
    case 'action_value':
      output_size = num_return_buckets
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS
    case 'state_value':
      output_size = num_return_buckets

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{model_name}',
  )
  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=predictor.initial_params(
          rng=jrandom.PRNGKey(1),
          targets=np.ones((1, 1), dtype=np.uint32),
      ),
      step=checkpoint_step,
  )
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )
  return neural_engines.ENGINE_FROM_POLICY[policy](
      return_buckets_values=return_buckets_values,
      predict_fn=neural_engines.wrap_predict_fn(
          predictor=predictor,
          params=params,
          batch_size=1,
      ),
  )


def _build_selfplay_engine(base_model: str, iteration: int | str, training_type: str = 'selfplay'):
  """Builds a fine-tuned neural engine.

  Args:
    base_model: Base model name (e.g., '9M', '136M', '270M').
    iteration: Training iteration checkpoint to load (int or str like 'best', '10000pairs').
    training_type: Type of training ('selfplay', 'lichess', etc.).

  Returns:
    Neural engine with fine-tuned parameters.
  """
  model_name = f'{base_model}_{training_type}'

  # Same architecture as base model
  if base_model == '9M':
    num_layers, embedding_dim, num_heads = 8, 256, 8
  elif base_model == '136M':
    num_layers, embedding_dim, num_heads = 8, 1024, 8
  else:  # 270M
    num_layers, embedding_dim, num_heads = 16, 1024, 8

  num_return_buckets = 128
  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=num_return_buckets,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)
  checkpoint_dir = os.path.join(os.getcwd(), f'../checkpoints/{model_name}')

  # Load from specific iteration, use EMA params
  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=predictor.initial_params(
          rng=jrandom.PRNGKey(1),
          targets=np.ones((1, 1), dtype=np.uint32),
      ),
      step=iteration,
      use_ema_params=True,
  )

  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )
  return neural_engines.ActionValueEngine(
      return_buckets_values=return_buckets_values,
      predict_fn=neural_engines.wrap_predict_fn(
          predictor=predictor,
          params=params,
          batch_size=1,
      ),
  )


ENGINE_BUILDERS = {
    'local': functools.partial(_build_neural_engine, model_name='local'),
    '9M': functools.partial(
        _build_neural_engine, model_name='9M', checkpoint_step=6_400_000
    ),
    '136M': functools.partial(
        _build_neural_engine, model_name='136M', checkpoint_step=6_400_000
    ),
    '270M': functools.partial(
        _build_neural_engine, model_name='270M', checkpoint_step=6_400_000
    ),
    # Selfplay-trained models (9M - iterations 0-20)
    '9M_selfplay': lambda: _build_selfplay_engine('9M', iteration=1),
    '9M_selfplay_iter0': lambda: _build_selfplay_engine('9M', iteration=0),  # Warm-up checkpoint
    '9M_selfplay_iter1': lambda: _build_selfplay_engine('9M', iteration=1),
    '9M_selfplay_iter2': lambda: _build_selfplay_engine('9M', iteration=2),
    '9M_selfplay_iter3': lambda: _build_selfplay_engine('9M', iteration=3),
    '9M_selfplay_iter4': lambda: _build_selfplay_engine('9M', iteration=4),
    '9M_selfplay_iter5': lambda: _build_selfplay_engine('9M', iteration=5),
    '9M_selfplay_iter6': lambda: _build_selfplay_engine('9M', iteration=6),
    '9M_selfplay_iter7': lambda: _build_selfplay_engine('9M', iteration=7),
    '9M_selfplay_iter8': lambda: _build_selfplay_engine('9M', iteration=8),
    '9M_selfplay_iter9': lambda: _build_selfplay_engine('9M', iteration=9),
    '9M_selfplay_iter10': lambda: _build_selfplay_engine('9M', iteration=10),
    '9M_selfplay_iter11': lambda: _build_selfplay_engine('9M', iteration=11),
    '9M_selfplay_iter12': lambda: _build_selfplay_engine('9M', iteration=12),
    '9M_selfplay_iter13': lambda: _build_selfplay_engine('9M', iteration=13),
    '9M_selfplay_iter14': lambda: _build_selfplay_engine('9M', iteration=14),
    '9M_selfplay_iter15': lambda: _build_selfplay_engine('9M', iteration=15),
    '9M_selfplay_iter16': lambda: _build_selfplay_engine('9M', iteration=16),
    '9M_selfplay_iter17': lambda: _build_selfplay_engine('9M', iteration=17),
    '9M_selfplay_iter18': lambda: _build_selfplay_engine('9M', iteration=18),
    '9M_selfplay_iter19': lambda: _build_selfplay_engine('9M', iteration=19),
    '9M_selfplay_iter20': lambda: _build_selfplay_engine('9M', iteration=20),
    # Selfplay-trained models (136M and 270M)
    '136M_selfplay': lambda: _build_selfplay_engine('136M', iteration=1),
    '136M_selfplay_iter1': lambda: _build_selfplay_engine('136M', iteration=1),
    '136M_selfplay_iter2': lambda: _build_selfplay_engine('136M', iteration=2),
    '136M_selfplay_iter3': lambda: _build_selfplay_engine('136M', iteration=3),
    '136M_selfplay_iter4': lambda: _build_selfplay_engine('136M', iteration=4),
    '136M_selfplay_iter5': lambda: _build_selfplay_engine('136M', iteration=5),
    '270M_selfplay': lambda: _build_selfplay_engine('270M', iteration=1),
    '270M_selfplay_iter1': lambda: _build_selfplay_engine('270M', iteration=1),
    '270M_selfplay_iter2': lambda: _build_selfplay_engine('270M', iteration=2),
    '270M_selfplay_iter3': lambda: _build_selfplay_engine('270M', iteration=3),
    '270M_selfplay_iter4': lambda: _build_selfplay_engine('270M', iteration=4),
    '270M_selfplay_iter5': lambda: _build_selfplay_engine('270M', iteration=5),
    # Lichess DPO-trained models (9M) - checkpoints named by pairs trained
    '9M_lichess_100k': lambda: _build_selfplay_engine('9M', iteration='100000', training_type='lichess'),
    '9M_lichess_200k': lambda: _build_selfplay_engine('9M', iteration='200000', training_type='lichess'),
    '9M_lichess_300k': lambda: _build_selfplay_engine('9M', iteration='300000', training_type='lichess'),
    '9M_lichess_400k': lambda: _build_selfplay_engine('9M', iteration='400000', training_type='lichess'),
    '9M_lichess_500k': lambda: _build_selfplay_engine('9M', iteration='500000', training_type='lichess'),
    '9M_lichess_600k': lambda: _build_selfplay_engine('9M', iteration='600000', training_type='lichess'),
    '9M_lichess_700k': lambda: _build_selfplay_engine('9M', iteration='700000', training_type='lichess'),
    '9M_lichess_800k': lambda: _build_selfplay_engine('9M', iteration='800000', training_type='lichess'),
    '9M_lichess_900k': lambda: _build_selfplay_engine('9M', iteration='900000', training_type='lichess'),
    '9M_lichess_1000k': lambda: _build_selfplay_engine('9M', iteration='1000000', training_type='lichess'),
    # Legacy/old iteration names (kept for backwards compatibility)
    '9M_lichess_best': lambda: _build_selfplay_engine('9M', iteration='best', training_type='lichess'),
    '9M_lichess': lambda: _build_selfplay_engine('9M', iteration=24, training_type='lichess'),
    '9M_lichess_iter2': lambda: _build_selfplay_engine('9M', iteration=2, training_type='lichess'),
    '9M_lichess_iter3': lambda: _build_selfplay_engine('9M', iteration=3, training_type='lichess'),
    '9M_lichess_iter4': lambda: _build_selfplay_engine('9M', iteration=4, training_type='lichess'),
    '9M_lichess_iter5': lambda: _build_selfplay_engine('9M', iteration=5, training_type='lichess'),
    '9M_lichess_iter6': lambda: _build_selfplay_engine('9M', iteration=6, training_type='lichess'),
    '9M_lichess_iter7': lambda: _build_selfplay_engine('9M', iteration=7, training_type='lichess'),
    '9M_lichess_iter8': lambda: _build_selfplay_engine('9M', iteration=8, training_type='lichess'),
    '9M_lichess_iter9': lambda: _build_selfplay_engine('9M', iteration=9, training_type='lichess'),
    '9M_lichess_iter10': lambda: _build_selfplay_engine('9M', iteration=10, training_type='lichess'),
    '9M_lichess_iter11': lambda: _build_selfplay_engine('9M', iteration=11, training_type='lichess'),
    '9M_lichess_iter12': lambda: _build_selfplay_engine('9M', iteration=12, training_type='lichess'),
    '9M_lichess_iter13': lambda: _build_selfplay_engine('9M', iteration=13, training_type='lichess'),
    '9M_lichess_iter14': lambda: _build_selfplay_engine('9M', iteration=14, training_type='lichess'),
    '9M_lichess_iter15': lambda: _build_selfplay_engine('9M', iteration=15, training_type='lichess'),
    '9M_lichess_iter16': lambda: _build_selfplay_engine('9M', iteration=16, training_type='lichess'),
    '9M_lichess_iter17': lambda: _build_selfplay_engine('9M', iteration=17, training_type='lichess'),
    '9M_lichess_iter18': lambda: _build_selfplay_engine('9M', iteration=18, training_type='lichess'),
    '9M_lichess_iter19': lambda: _build_selfplay_engine('9M', iteration=19, training_type='lichess'),
    '9M_lichess_iter20': lambda: _build_selfplay_engine('9M', iteration=20, training_type='lichess'),
    '9M_lichess_iter21': lambda: _build_selfplay_engine('9M', iteration=21, training_type='lichess'),
    '9M_lichess_iter22': lambda: _build_selfplay_engine('9M', iteration=22, training_type='lichess'),
    '9M_lichess_iter23': lambda: _build_selfplay_engine('9M', iteration=23, training_type='lichess'),
    '9M_lichess_iter24': lambda: _build_selfplay_engine('9M', iteration=24, training_type='lichess'),
    'stockfish': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05)
    ),
    'stockfish_all_moves': lambda: stockfish_engine.AllMovesStockfishEngine(
        limit=chess.engine.Limit(time=0.05)
    ),
    'leela_chess_zero_depth_1': lambda: lc0_engine.AllMovesLc0Engine(
        limit=chess.engine.Limit(nodes=1),
    ),
    'leela_chess_zero_policy_net': lambda: lc0_engine.Lc0Engine(
        limit=chess.engine.Limit(nodes=1),
    ),
    'leela_chess_zero_400_sims': lambda: lc0_engine.Lc0Engine(
        limit=chess.engine.Limit(nodes=400),
    ),
}
