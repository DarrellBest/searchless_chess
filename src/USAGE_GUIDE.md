# Searchless Chess - Usage Guide

## Environment Setup

Before running any scripts, you must activate the conda environment and set the PYTHONPATH:

```bash
# Activate the conda environment
conda activate searchless_chess

# Set PYTHONPATH (CRITICAL - must be the parent directory)
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects

# Navigate to the src directory
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src
```

## Available Scripts

### 1. Puzzles Evaluation (puzzles.py)

Evaluates chess engines on Lichess puzzles to test their tactical ability.

**Available Agents:**
- Neural models: `9M`, `136M`, `270M`
- Stockfish: `stockfish`, `stockfish_all_moves`
- Leela Chess Zero: `leela_chess_zero_depth_1`, `leela_chess_zero_policy_net`, `leela_chess_zero_400_sims`
- Local trained model: `local`

**Basic Usage:**

```bash
# Test the 9M model on 10 puzzles
python puzzles.py --agent=9M --num_puzzles=10

# Test the 270M model on 50 puzzles
python puzzles.py --agent=270M --num_puzzles=50

# Test Stockfish on 20 puzzles
python puzzles.py --agent=stockfish --num_puzzles=20

# Test Leela Chess Zero policy network on 15 puzzles
python puzzles.py --agent=leela_chess_zero_policy_net --num_puzzles=15
```

**Example Output:**
```
{'puzzle_id': 0, 'correct': True, 'rating': 669}
{'puzzle_id': 1, 'correct': True, 'rating': 1932}
{'puzzle_id': 2, 'correct': False, 'rating': 2106}
```

**Performance Notes:**
- First run loads model checkpoints (takes ~2-10 seconds per model)
- Each puzzle evaluation takes 0.1-2 seconds depending on the agent
- Stockfish is fastest for evaluation
- Neural models require GPU for good performance

---

### 2. Tournament (tournament.py)

Runs a round-robin tournament between multiple engines to compute Elo ratings.

**Engines Included:**
- `9M`, `136M`, `270M` (neural models)
- `stockfish`, `stockfish_all_moves`
- `leela_chess_zero_depth_1`, `leela_chess_zero_policy_net`, `leela_chess_zero_400_sims`

**Basic Usage:**

```bash
# Run a small tournament with 2 games per engine pair
python tournament.py --num_games=2

# Run a larger tournament with 10 games per engine pair
python tournament.py --num_games=10

# Full tournament (as in paper) with 200 games per pair
python tournament.py --num_games=200
```

**What Happens:**
1. Loads all engine models and executables
2. Loads chess opening positions from ECO database
3. Plays games between all pairs of engines (round-robin)
4. For each pair, plays multiple games with different openings
5. Each game is played from both sides (white/black)
6. Saves all games to `../data/tournament_games.pgn`

**Tournament Size:**
- With 8 engines, there are 28 unique pairs
- With `--num_games=2`: plays 2×2 = 4 games per pair = 112 total games
- With `--num_games=10`: plays 10×2 = 20 games per pair = 560 total games
- With `--num_games=200`: plays 200×2 = 400 games per pair = 11,200 total games!

**Time Estimates:**
- `--num_games=2`: ~30-60 minutes
- `--num_games=10`: ~3-5 hours
- `--num_games=200`: ~2-3 days (continuous)

**Computing Elo Ratings:**

After the tournament completes, compute Elo ratings with BayesElo:

```bash
cd ../BayesElo
./bayeselo

# Inside BayesElo prompt:
ResultSet> readpgn ../data/tournament_games.pgn
ResultSet> elo
ResultSet-EloRating> mm
ResultSet-EloRating> exactdist
ResultSet-EloRating> ratings
# ... Elo ratings will be displayed ...
ResultSet-EloRating> x
ResultSet> x
```

---

### 3. Training (train.py)

Trains a transformer model on chess data.

**Available Policies:**
- `action_value` - Predicts Q-values for all legal moves
- `state_value` - Predicts the value of the current position
- `behavioral_cloning` - Learns to imitate Stockfish move choices

**Basic Usage:**

```bash
# Train with default settings
python train.py

# Train with specific policy
python train.py --policy=action_value
python train.py --policy=state_value
python train.py --policy=behavioral_cloning
```

**Notes:**
- Training requires the full dataset (not included in test download)
- Downloads the dataset with `cd ../data && ./download.sh`
- Training is GPU-intensive
- Checkpoints are saved to `../checkpoints/local/`

---

### 4. Jupyter Notebook (searchless_chess.ipynb)

Interactive analysis notebook for exploring model behavior.

**Usage:**

```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects

# Start Jupyter from the src directory
jupyter notebook

# In browser, open: searchless_chess.ipynb
```

**What's Inside:**
- Load and inspect trained models
- Visualize action-value predictions
- Analyze model behavior on specific positions
- Compute win percentages for legal moves
- Compare different model sizes

---

## Quick Reference Commands

```bash
# Setup (run once per session)
conda activate searchless_chess
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects
cd /mnt/c/Users/darre/PycharmProjects/searchless_chess/src

# Quick puzzle test (3 puzzles, fast)
python puzzles.py --agent=9M --num_puzzles=3

# Comprehensive puzzle evaluation (50 puzzles)
python puzzles.py --agent=270M --num_puzzles=50

# Mini tournament (quick test, ~30 min)
python tournament.py --num_games=2

# Full-scale tournament (reproduces paper results, ~2-3 days)
python tournament.py --num_games=200

# Start interactive notebook
jupyter notebook
```

---

## Troubleshooting

### Module Not Found Error

If you see `ModuleNotFoundError: No module named 'searchless_chess'`:

**Solution:** Set PYTHONPATH to the parent directory:
```bash
export PYTHONPATH=/mnt/c/Users/darre/PycharmProjects
```

### CUDA Out of Memory

If you see CUDA memory errors during tournament:

**Solution:** The tournament loads all models at once. Try:
- Reducing number of games
- Running on a machine with more GPU memory
- Running only subset of engines (modify tournament.py)

### Slow Performance

If puzzles or games are very slow:

**Check:**
- Is GPU being used? Run: `nvidia-smi`
- Is conda environment activated?
- Are you using the correct PYTHONPATH?

---

## Model Checkpoints

**Location:** `../checkpoints/`

- **9M model:** 34 MB, ~9 million parameters
  - Fast inference, good for testing
  
- **136M model:** 520 MB, ~136 million parameters
  - Balanced performance/speed
  
- **270M model:** 1 GB, ~270 million parameters
  - Best performance, slowest inference

Each model directory contains:
- `params/` - Model weights
- `params_ema/` - Exponential moving average weights (typically better)

---

## Dataset Information

**Location:** `../data/`

**Downloaded Files:**
- `eco_openings.pgn` - Chess opening database (500 KB)
- `puzzles.csv` - Lichess puzzles (4.7 MB)
- `test/` - Test datasets (141 MB)

**Full Training Data (not downloaded by default):**
- `train/action_value_data.bag` - 1.1 TB (2148 sharded files)
- `train/behavioral_cloning_data.bag` - 34 GB
- `train/state_value_data.bag` - 36 GB

To download full training data:
```bash
cd ../data
./download.sh  # WARNING: Downloads >1 TB
```

---

## External Engines

**Stockfish:**
- Path: `../Stockfish/src/stockfish`
- Version: Latest (Oct 2024)
- Test: `echo "uci" | ../Stockfish/src/stockfish`

**Leela Chess Zero:**
- Path: `../lc0/build/release/lc0`
- Network: `../lc0/build/release/768x15x24h-t82-swa-7464000.pb`
- Version: v0.30.0
- Test: `echo "uci" | ../lc0/build/release/lc0`

**BayesElo:**
- Path: `../BayesElo/bayeselo`
- Purpose: Compute Elo ratings from tournament PGN files
- Test: `echo "quit" | ../BayesElo/bayeselo`

