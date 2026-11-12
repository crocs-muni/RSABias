"""
Summarizes the reports of the classify task: the top group of every key, and the combined verdict for all keys
assuming they come from the same source. The combined score of a group is the product of its per-key scores, as the
evaluate task computes it for batches of keys, summed as logarithms so that many keys do not underflow.

Usage: python tools/summarize_classification.py reports/*/key_classification.json
"""
import collections
import json
import math
import sys


def main(paths, top=3):
    reports = {}
    for path in paths:
        with open(path) as f:
            reports[path] = json.load(f)
    groups = next(iter(reports.values()))

    print('Top group per key:')
    for path, report in reports.items():
        best = max(report, key=lambda g: report[g]['score'])
        print('  {}: group {} (score {:.4g})'.format(path, best, report[best]['score']))
    counts = collections.Counter(max(r, key=lambda g: r[g]['score']) for r in reports.values())
    print('Top-group counts:', dict(counts.most_common()))

    # A zero score means the key is impossible for the group, which excludes the group altogether
    log_scores = {g: sum(math.log(r[g]['score']) if r[g]['score'] > 0 else -math.inf for r in reports.values())
                  for g in groups}
    ranked = sorted(log_scores.items(), key=lambda item: item[1], reverse=True)
    print('\nCombined over {} keys (same source assumed):'.format(len(reports)))
    possible = [(g, s) for g, s in ranked if s > -math.inf]
    if not possible:
        print('  no group is possible for all keys; they likely come from several sources')
    for group, score in possible[:top]:
        print('  group {}: log-score {:.2f}'.format(group, score))
        for source in groups[group]['sources']:
            print('      {}'.format(source))
    print('  excluded (some key impossible for them): {} of {} groups'.format(len(ranked) - len(possible),
                                                                             len(ranked)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
