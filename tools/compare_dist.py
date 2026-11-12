"""
Compares two key distributions written by the dist task: per feature, the share of every value in each set, the
difference in percentage points and a chi-square test; then the feature combinations that differ most and those seen
in only one set.

Each side is a dist.json file, or a directory whose dist.json files (e.g. several key lengths, or all sources of one
group in rsabias/model/distributions) are summed. Both sides must use the same transformation.

Usage: python tools/compare_dist.py A B [--top 10]
"""
import argparse
import ast
import collections
import json
import os
import sys

from scipy.stats import chi2_contingency

# Feature names of the model's transformation (rsabias/model/transformations/5p_5q_blum_mod_roca.json)
MODEL_FEATURES = ['msb5 of p', 'msb5 of q', 'Blum (p, q = 3 mod 4)', 'modular fingerprint', 'ROCA']


def load(path):
    """Returns (transformation name, Counter of feature tuples) for a dist.json or a directory of them."""
    if os.path.isdir(path):
        files = sorted(os.path.join(root, f) for root, _, names in os.walk(path) for f in names if f == 'dist.json')
    else:
        files = [path]
    if not files:
        sys.exit('No dist.json in {}'.format(path))
    name = None
    counts = collections.Counter()
    for file in files:
        with open(file) as f:
            dist = json.load(f)
        if name is not None and dist['name'] != name:
            sys.exit('{} uses a different transformation than the other files'.format(file))
        name = dist['name']
        for cell, count in dist['counts'].items():
            counts[ast.literal_eval(cell)] += count
    return name, counts


def chi_square(a, b):
    """p-value of the chi-square test that both count vectors come from one distribution."""
    keys = [k for k in set(a) | set(b) if a[k] + b[k] > 0]
    if len(keys) < 2:
        return 1.0
    return chi2_contingency([[a[k] for k in keys], [b[k] for k in keys]])[1]


def share(counter, key, total):
    return 100.0 * counter[key] / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description='Compare two key distributions written by the dist task')
    parser.add_argument('a', help='dist.json, or a directory whose dist.json files are summed')
    parser.add_argument('b', help='dist.json, or a directory whose dist.json files are summed')
    parser.add_argument('--top', type=int, default=10, help='number of feature combinations to list')
    args = parser.parse_args()

    name_a, a = load(args.a)
    name_b, b = load(args.b)
    if name_a != name_b:
        sys.exit('The two sides use different transformations:\n  {}\n  {}'.format(name_a, name_b))
    total_a, total_b = sum(a.values()), sum(b.values())
    dims = len(next(iter(a)))
    names = MODEL_FEATURES if dims == len(MODEL_FEATURES) and name_a.startswith('Combine([MostSignificantBits({p},5') \
        else ['feature {}'.format(i + 1) for i in range(dims)]

    print('A: {} keys ({})'.format(total_a, args.a))
    print('B: {} keys ({})'.format(total_b, args.b))

    # Per feature: the marginal distribution of each side
    for i, feature in enumerate(names):
        ma, mb = collections.Counter(), collections.Counter()
        for cell, count in a.items():
            ma[cell[i]] += count
        for cell, count in b.items():
            mb[cell[i]] += count
        print('\n{} (chi-square p = {:.3g})'.format(feature, chi_square(ma, mb)))
        print('  {:>8} {:>8} {:>8} {:>9}'.format('value', 'A %', 'B %', 'B-A pp'))
        for value in sorted(set(ma) | set(mb), key=lambda v: (type(v).__name__, v)):
            sa, sb = share(ma, value, total_a), share(mb, value, total_b)
            print('  {:>8} {:8.2f} {:8.2f} {:+9.2f}'.format(str(value), sa, sb, sb - sa))

    # All features together
    cells = set(a) | set(b)
    deltas = {c: share(b, c, total_b) - share(a, c, total_a) for c in cells}
    tvd = sum(abs(d) for d in deltas.values()) / 2
    print('\nAll features together: {} combinations, total variation distance {:.2f} pp, chi-square p = {:.3g}'
          .format(len(cells), tvd, chi_square(a, b)))
    print('  Largest differences (B-A pp):')
    for cell in sorted(cells, key=lambda c: abs(deltas[c]), reverse=True)[:args.top]:
        print('    {} {:+.2f}  (A {}, B {})'.format(cell, deltas[cell], a[cell], b[cell]))
    only_a, only_b = [c for c in cells if not b[c]], [c for c in cells if not a[c]]
    print('  Only in A: {} combinations, {} keys'.format(len(only_a), sum(a[c] for c in only_a)))
    print('  Only in B: {} combinations, {} keys'.format(len(only_b), sum(b[c] for c in only_b)))
    print('\nWith many keys even negligible differences are significant (small p); judge them by the pp columns.')


if __name__ == '__main__':
    main()
