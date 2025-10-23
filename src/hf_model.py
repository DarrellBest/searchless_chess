"""HuggingFace model wrapper for searchless chess."""

import json
import os
from typing import Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from searchless_chess.src import tokenizer
from searchless_chess.src import transformer
from searchless_chess.src import utils


class SearchlessChessConfig:
    """Configuration for SearchlessChess model."""

    def __init__(
        self,
        vocab_size: int = 1968,
        output_size: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        max_sequence_length: int = 79,
        num_return_buckets: int = 128,
        model_name: str = "9M",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.num_return_buckets = num_return_buckets
        self.model_name = model_name

        # Store any extra kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "output_size": self.output_size,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_sequence_length": self.max_sequence_length,
            "num_return_buckets": self.num_return_buckets,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SearchlessChessConfig":
        """Load config from dictionary."""
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str):
        """Save config to directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SearchlessChessConfig":
        """Load config from directory."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class SearchlessChessModel:
    """HuggingFace-compatible wrapper for SearchlessChess JAX/Haiku model."""

    def __init__(self, config: SearchlessChessConfig):
        self.config = config

        # Build transformer config
        self.transformer_config = transformer.TransformerConfig(
            vocab_size=config.vocab_size,
            output_size=config.output_size,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=config.max_sequence_length,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            embedding_dim=config.embedding_dim,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )

        # Build predictor
        self.predictor = transformer.build_transformer_predictor(self.transformer_config)

        # Initialize params
        self.params = None
        self.return_buckets_values = None

        # Get return bucket values
        _, self.return_buckets_values = utils.get_uniform_buckets_edges_values(
            config.num_return_buckets
        )

    def load_params(self, params_path: str):
        """Load parameters from Orbax checkpoint."""
        # Convert to absolute path (Orbax requires absolute paths)
        params_path = os.path.abspath(params_path)

        # Create dummy params for structure
        dummy_params = self.predictor.initial_params(
            rng=jax.random.PRNGKey(0),
            targets=np.ones((1, 1), dtype=np.uint32),
        )

        # Load checkpoint
        restore_args = ocp.checkpoint_utils.construct_restore_args(dummy_params)
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self.params = checkpointer.restore(params_path, restore_args=restore_args)

    def predict(self, fen: str, temperature: float = 1.0) -> Dict:
        """Predict move from FEN position.

        Args:
            fen: Chess position in FEN notation
            temperature: Temperature for sampling (1.0 = no modification)

        Returns:
            Dictionary with:
                - q_values: Q-value distribution
                - action_probs: Action probabilities
                - best_action: Best action index
                - best_move: Best move in UCI notation
        """
        if self.params is None:
            raise ValueError("Model parameters not loaded. Call load_params() first.")

        # Tokenize input
        tokens = tokenizer.tokenize(fen)
        tokens = tokens[None, :]  # Add batch dimension

        # Get predictions
        bucket_log_probs = self.predictor.predict(
            params=self.params,
            targets=tokens,
            rng=None,
        )

        # Extract action Q-values (second to last position)
        action_bucket_log_probs = bucket_log_probs[0, -2]  # [num_return_buckets]
        action_bucket_probs = jnp.exp(action_bucket_log_probs)

        # Compute Q-value for each action bucket
        q_value = float(jnp.dot(action_bucket_probs, self.return_buckets_values))

        # Get action probabilities from Q-values
        # Use softmax over return bucket expectations
        action_values = jnp.dot(
            jnp.exp(bucket_log_probs[0, -2:]),
            self.return_buckets_values,
        )

        # Apply temperature and softmax
        action_logits = action_values / temperature
        action_probs = jax.nn.softmax(action_logits)

        # Get best action
        best_action = int(jnp.argmax(action_probs))

        # Convert action to move
        best_move = utils.ACTION_TO_MOVE.get(best_action, "unknown")

        return {
            "q_value": q_value,
            "action_probs": np.array(action_probs),
            "best_action": best_action,
            "best_move": best_move,
        }

    def save_pretrained(self, save_directory: str):
        """Save model to directory in HuggingFace format."""
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save parameters as numpy arrays
        if self.params is not None:
            params_cpu = jax.device_get(self.params)
            params_flat, tree_def = jax.tree.flatten(params_cpu)

            # Save flattened params
            params_path = os.path.join(save_directory, "params.npz")
            np.savez(params_path, *params_flat)

            # Save tree structure
            import pickle
            tree_path = os.path.join(save_directory, "tree_structure.pkl")
            with open(tree_path, "wb") as f:
                pickle.dump(tree_def, f)

        # Copy necessary source files for standalone usage
        import shutil
        src_dir = os.path.dirname(__file__)
        code_dir = os.path.join(save_directory, "searchless_chess_code")
        os.makedirs(code_dir, exist_ok=True)

        # Copy core modules and fix imports for standalone usage
        def fix_imports(content):
            """Replace absolute imports with relative imports."""
            content = content.replace("from searchless_chess.src import tokenizer", "import tokenizer")
            content = content.replace("from searchless_chess.src import transformer", "import transformer")
            content = content.replace("from searchless_chess.src import utils", "import utils")
            content = content.replace("from searchless_chess.src import constants", "import constants")
            content = content.replace("from searchless_chess.src import config as config_lib", "import config as config_lib")
            content = content.replace("from searchless_chess.src import config", "import config")
            return content

        for module in ["tokenizer.py", "transformer.py", "constants.py", "utils.py", "config.py"]:
            src_file = os.path.join(src_dir, module)
            dst_file = os.path.join(code_dir, module)
            if os.path.exists(src_file):
                with open(src_file, 'r') as f:
                    content = fix_imports(f.read())
                with open(dst_file, 'w') as f:
                    f.write(content)

        # Create standalone hf_model.py
        standalone_hf_model = os.path.join(code_dir, "hf_model.py")
        with open(__file__, 'r') as source:
            content = fix_imports(source.read())
            with open(standalone_hf_model, 'w') as dest:
                dest.write(content)

        # Create __init__.py
        with open(os.path.join(code_dir, "__init__.py"), "w") as f:
            f.write("# Searchless Chess code bundle\n")

        # Save model info
        model_info = {
            "model_type": "searchless_chess",
            "framework": "jax",
            "library": "dm-haiku",
            "includes_source": True,
            "source_modules": ["tokenizer.py", "transformer.py", "constants.py", "utils.py", "config.py"],
        }
        with open(os.path.join(save_directory, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SearchlessChessModel":
        """Load model from directory."""
        # Load config
        config = SearchlessChessConfig.from_pretrained(model_path)

        # Create model
        model = cls(config)

        # Load parameters
        params_path = os.path.join(model_path, "params.npz")
        tree_path = os.path.join(model_path, "tree_structure.pkl")

        if os.path.exists(params_path) and os.path.exists(tree_path):
            # Load tree structure
            import pickle
            with open(tree_path, "rb") as f:
                tree_def = pickle.load(f)

            # Load params
            params_data = np.load(params_path)
            params_flat = [params_data[f"arr_{i}"] for i in range(len(params_data.files))]

            # Reconstruct pytree
            model.params = jax.tree.unflatten(tree_def, params_flat)

        return model


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "9M",
    use_ema: bool = True,
) -> SearchlessChessModel:
    """Create HuggingFace model from existing checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., checkpoints/9M_selfplay/4)
        model_name: Model size (9M, 136M, 270M)
        use_ema: Whether to load EMA parameters

    Returns:
        SearchlessChessModel ready to save or use
    """
    # Determine architecture from model name
    if model_name == "9M":
        num_layers, embedding_dim, num_heads = 8, 256, 8
    elif model_name == "136M":
        num_layers, embedding_dim, num_heads = 8, 1024, 8
    else:  # 270M
        num_layers, embedding_dim, num_heads = 16, 1024, 8

    # Create config
    config = SearchlessChessConfig(
        vocab_size=1968,
        output_size=128,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_sequence_length=79,
        num_return_buckets=128,
        model_name=model_name,
    )

    # Create model
    model = SearchlessChessModel(config)

    # Load parameters from Orbax checkpoint
    params_dir = "params_ema" if use_ema else "params"
    params_path = os.path.join(checkpoint_path, params_dir)
    model.load_params(params_path)

    return model
