"""Convert searchless chess checkpoint to HuggingFace format."""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

from searchless_chess.src import hf_model


_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    '../checkpoints/9M_selfplay/4',
    'Path to checkpoint directory',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    '9M',
    'Model size (9M, 136M, 270M)',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '../hf_models/9M_selfplay',
    'Output directory for HuggingFace model',
)

_USE_EMA = flags.DEFINE_boolean(
    'use_ema',
    True,
    'Use EMA parameters',
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info(f'Converting checkpoint: {_CHECKPOINT_PATH.value}')
    logging.info(f'Model name: {_MODEL_NAME.value}')
    logging.info(f'Use EMA: {_USE_EMA.value}')
    logging.info(f'Output directory: {_OUTPUT_DIR.value}')

    # Create model from checkpoint
    model = hf_model.create_model_from_checkpoint(
        checkpoint_path=_CHECKPOINT_PATH.value,
        model_name=_MODEL_NAME.value,
        use_ema=_USE_EMA.value,
    )

    # Save in HuggingFace format
    model.save_pretrained(_OUTPUT_DIR.value)

    logging.info(f'Model saved to: {_OUTPUT_DIR.value}')
    logging.info('Files created:')
    logging.info('  - config.json')
    logging.info('  - params.npz')
    logging.info('  - tree_structure.pkl')
    logging.info('  - model_info.json')

    # Test loading
    logging.info('\nTesting model loading...')
    loaded_model = hf_model.SearchlessChessModel.from_pretrained(_OUTPUT_DIR.value)

    # Test prediction
    test_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    result = loaded_model.predict(test_fen)

    logging.info(f'Test prediction successful!')
    logging.info(f'  Best move: {result["best_move"]}')
    logging.info(f'  Q-value: {result["q_value"]:.4f}')


if __name__ == '__main__':
    app.run(main)
