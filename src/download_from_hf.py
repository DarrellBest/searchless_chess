"""Download and use model from HuggingFace Hub."""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging


_REPO_ID = flags.DEFINE_string(
    'repo_id',
    None,
    'HuggingFace repository ID (e.g., username/searchless-chess-9M-selfplay)',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '../hf_models/downloaded',
    'Directory to download model to',
)

_TEST_FEN = flags.DEFINE_string(
    'test_fen',
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'FEN position to test with',
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if _REPO_ID.value is None:
        raise app.UsageError('--repo_id is required')

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            'huggingface_hub is required. Install with: pip install huggingface_hub'
        )

    from searchless_chess.src import hf_model

    logging.info(f'Downloading model from: {_REPO_ID.value}')

    # Download model
    try:
        model_path = snapshot_download(
            repo_id=_REPO_ID.value,
            local_dir=_OUTPUT_DIR.value,
        )
        logging.info(f'Model downloaded to: {model_path}')
    except Exception as e:
        logging.error(f'Failed to download: {e}')
        raise

    # Load model
    logging.info('Loading model...')
    model = hf_model.SearchlessChessModel.from_pretrained(_OUTPUT_DIR.value)
    logging.info('Model loaded successfully!')

    # Test prediction
    logging.info(f'\nTesting with position: {_TEST_FEN.value}')
    result = model.predict(_TEST_FEN.value)

    logging.info('\nPrediction results:')
    logging.info(f'  Best move: {result["best_move"]}')
    logging.info(f'  Q-value: {result["q_value"]:.4f}')
    logging.info(f'  Best action index: {result["best_action"]}')


if __name__ == '__main__':
    app.run(main)
