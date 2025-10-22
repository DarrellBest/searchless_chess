"""Evaluates and compares base vs selfplay-trained models."""

import json
import os
import subprocess
import sys
from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

_BASE_MODEL = flags.DEFINE_enum(
    'base_model',
    '9M',
    ['9M', '136M', '270M'],
    'The base model to evaluate.',
)

_ITERATION = flags.DEFINE_integer(
    'iteration',
    1,
    'Training iteration to evaluate.',
)

_NUM_PUZZLES = flags.DEFINE_integer(
    'num_puzzles',
    100,
    'Number of puzzles to evaluate on.',
)


def run_puzzle_evaluation(agent: str, num_puzzles: int) -> dict:
  """Runs puzzle evaluation and returns results.

  Args:
    agent: Agent name (e.g., '9M' or '9M_selfplay').
    num_puzzles: Number of puzzles to evaluate.

  Returns:
    Dict with accuracy and rating breakdown.
  """
  logging.info(f'Evaluating {agent} on {num_puzzles} puzzles...')

  # Run puzzles.py and capture output
  result = subprocess.run(
      ['python', 'puzzles.py', f'--agent={agent}', f'--num_puzzles={num_puzzles}'],
      capture_output=True,
      text=True,
  )

  if result.returncode != 0:
    logging.error(f'Puzzle evaluation failed for {agent}')
    logging.error(result.stderr)
    return None

  # Parse output to extract results
  lines = result.stdout.strip().split('\n')
  results = []

  for line in lines:
    if line.startswith('{'):
      try:
        puzzle_result = eval(line)
        results.append(puzzle_result)
      except:
        pass

  if not results:
    logging.warning(f'No results parsed for {agent}')
    return None

  # Calculate statistics
  correct = sum(1 for r in results if r.get('correct', False))
  total = len(results)
  accuracy = correct / total if total > 0 else 0.0

  # Rating breakdown
  rating_ranges = {
      '0-1000': (0, 1000),
      '1000-1500': (1000, 1500),
      '1500-2000': (1500, 2000),
      '2000-2500': (2000, 2500),
      '2500+': (2500, 10000),
  }

  range_stats = {}
  for range_name, (min_rating, max_rating) in rating_ranges.items():
    range_puzzles = [
        r for r in results
        if min_rating <= r.get('rating', 0) < max_rating
    ]
    if range_puzzles:
      range_correct = sum(1 for r in range_puzzles if r.get('correct', False))
      range_stats[range_name] = {
          'total': len(range_puzzles),
          'correct': range_correct,
          'accuracy': range_correct / len(range_puzzles),
      }

  return {
      'agent': agent,
      'total_puzzles': total,
      'correct': correct,
      'accuracy': accuracy,
      'rating_breakdown': range_stats,
  }


def compare_results(base_results: dict, selfplay_results: dict):
  """Prints comparison between base and selfplay results.

  Args:
    base_results: Results from base model.
    selfplay_results: Results from selfplay model.
  """
  print('\n' + '=' * 80)
  print('EVALUATION RESULTS COMPARISON')
  print('=' * 80)

  print(f'\nBase Model: {base_results["agent"]}')
  print(f'Accuracy: {base_results["accuracy"]:.2%} ({base_results["correct"]}/{base_results["total_puzzles"]})')

  print(f'\nSelfplay Model: {selfplay_results["agent"]}')
  print(f'Accuracy: {selfplay_results["accuracy"]:.2%} ({selfplay_results["correct"]}/{selfplay_results["total_puzzles"]})')

  # Improvement
  accuracy_improvement = selfplay_results["accuracy"] - base_results["accuracy"]
  puzzle_improvement = selfplay_results["correct"] - base_results["correct"]

  print(f'\nImprovement:')
  print(f'  Accuracy: {accuracy_improvement:+.2%}')
  print(f'  Additional Puzzles Solved: {puzzle_improvement:+d}')

  # Rating breakdown comparison
  print(f'\nRating Breakdown:')
  print(f'{"Range":<15} {"Base Acc.":<12} {"Selfplay Acc.":<15} {"Improvement":<12}')
  print('-' * 80)

  all_ranges = set(base_results['rating_breakdown'].keys()) | set(selfplay_results['rating_breakdown'].keys())
  for range_name in sorted(all_ranges):
    base_acc = base_results['rating_breakdown'].get(range_name, {}).get('accuracy', 0.0)
    selfplay_acc = selfplay_results['rating_breakdown'].get(range_name, {}).get('accuracy', 0.0)
    improvement = selfplay_acc - base_acc

    print(f'{range_name:<15} {base_acc:<12.2%} {selfplay_acc:<15.2%} {improvement:+.2%}')

  print('=' * 80)

  # Save results to file
  results_file = f'../data/{base_results["agent"]}_vs_{selfplay_results["agent"]}_comparison.json'
  with open(results_file, 'w') as f:
    json.dump({
        'base': base_results,
        'selfplay': selfplay_results,
        'improvement': {
            'accuracy': accuracy_improvement,
            'puzzles': puzzle_improvement,
        }
    }, f, indent=2)

  print(f'\nResults saved to: {results_file}')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  base_model = _BASE_MODEL.value
  selfplay_model = f'{base_model}_selfplay'

  # Check if selfplay checkpoint exists
  checkpoint_dir = os.path.join(
      os.getcwd(),
      f'../checkpoints/{selfplay_model}',
  )
  iteration_dir = os.path.join(checkpoint_dir, str(_ITERATION.value))

  if not os.path.exists(iteration_dir):
    logging.error(f'Selfplay checkpoint not found: {iteration_dir}')
    logging.error('Please train the model first using selfplay_train.py')
    sys.exit(1)

  # Evaluate base model
  logging.info(f'Starting evaluation comparison for {base_model}')
  base_results = run_puzzle_evaluation(base_model, _NUM_PUZZLES.value)

  if base_results is None:
    logging.error('Base model evaluation failed')
    sys.exit(1)

  # Evaluate selfplay model
  selfplay_results = run_puzzle_evaluation(selfplay_model, _NUM_PUZZLES.value)

  if selfplay_results is None:
    logging.error('Selfplay model evaluation failed')
    sys.exit(1)

  # Compare results
  compare_results(base_results, selfplay_results)


if __name__ == '__main__':
  app.run(main)
