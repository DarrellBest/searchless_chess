#!/usr/bin/env python3
"""Analyzes puzzle results by rating buckets."""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_results_file(filepath: str) -> List[Dict]:
    """Parse JSON lines from a puzzle results file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    result = eval(line)  # Safe here since we control the input
                    results.append(result)
                except:
                    pass
    return results


def analyze_by_rating_buckets(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze results by rating buckets.

    Returns:
        Dict mapping bucket name to stats (total, correct, accuracy)
    """
    # Define rating buckets
    buckets = [
        ('<1500', 0, 1500),
        ('1500-2000', 1500, 2000),
        ('2000-2500', 2000, 2500),
        ('2500+', 2500, 10000),
    ]

    bucket_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for result in results:
        rating = result['rating']
        correct = result['correct']

        # Find which bucket this puzzle belongs to
        for bucket_name, min_rating, max_rating in buckets:
            if min_rating <= rating < max_rating:
                bucket_stats[bucket_name]['total'] += 1
                if correct:
                    bucket_stats[bucket_name]['correct'] += 1
                break

    # Calculate accuracies
    for bucket_name in bucket_stats:
        total = bucket_stats[bucket_name]['total']
        correct = bucket_stats[bucket_name]['correct']
        if total > 0:
            bucket_stats[bucket_name]['accuracy'] = correct / total
        else:
            bucket_stats[bucket_name]['accuracy'] = 0.0

    return dict(bucket_stats)


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_puzzle_results.py <results_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    results = parse_results_file(filepath)

    if not results:
        print(f"No valid results found in {filepath}")
        sys.exit(1)

    # Overall stats
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    overall_accuracy = correct / total if total > 0 else 0

    # By rating bucket
    bucket_stats = analyze_by_rating_buckets(results)

    # Output in a format the bash script can parse
    print(f"TOTAL:{total}")
    print(f"CORRECT:{correct}")
    print(f"ACCURACY:{overall_accuracy:.3f}")

    for bucket_name in ['<1500', '1500-2000', '2000-2500', '2500+']:
        if bucket_name in bucket_stats:
            stats = bucket_stats[bucket_name]
            print(f"BUCKET:{bucket_name}:{stats['correct']}/{stats['total']}:{stats['accuracy']:.3f}")
        else:
            print(f"BUCKET:{bucket_name}:0/0:0.000")


if __name__ == '__main__':
    main()
