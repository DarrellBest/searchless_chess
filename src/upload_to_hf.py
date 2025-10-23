"""Upload model to HuggingFace Hub."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging


_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    '../hf_models/9M_selfplay',
    'Directory containing HuggingFace model',
)

_REPO_NAME = flags.DEFINE_string(
    'repo_name',
    'searchless-chess-9M-selfplay',
    'Name for HuggingFace repository',
)

_USERNAME = flags.DEFINE_string(
    'username',
    None,
    'HuggingFace username (required)',
)

_PRIVATE = flags.DEFINE_boolean(
    'private',
    False,
    'Create private repository',
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if _USERNAME.value is None:
        raise app.UsageError('--username is required')

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        raise ImportError(
            'huggingface_hub is required. Install with: pip install huggingface_hub'
        )

    logging.info(f'Uploading model from: {_MODEL_DIR.value}')
    logging.info(f'Repository: {_USERNAME.value}/{_REPO_NAME.value}')

    # Create repository
    repo_id = f'{_USERNAME.value}/{_REPO_NAME.value}'
    try:
        create_repo(
            repo_id=repo_id,
            private=_PRIVATE.value,
            exist_ok=True,
        )
        logging.info(f'Repository created/verified: {repo_id}')
    except Exception as e:
        logging.error(f'Failed to create repository: {e}')
        raise

    # Upload model files
    try:
        upload_folder(
            folder_path=_MODEL_DIR.value,
            repo_id=repo_id,
            commit_message='Upload searchless chess model',
        )
        logging.info('Upload successful!')
        logging.info(f'Model available at: https://huggingface.co/{repo_id}')
    except Exception as e:
        logging.error(f'Failed to upload: {e}')
        raise


if __name__ == '__main__':
    app.run(main)
