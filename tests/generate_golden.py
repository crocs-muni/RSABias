"""
Records the golden outputs the tests compare against. Run it only in the legacy reference image, whose outputs are
the ground truth for later dependency upgrades. Current code no longer runs on its Python 3.7, so run it in a checkout
of the tag esorics2020-env (see README) and copy tests/data and tests/golden over:

    git worktree add ../rsabias-legacy esorics2020-env && cd ../rsabias-legacy
    docker run --rm -v "$PWD:/opt/rsabias" --entrypoint python rsabias:legacy tests/generate_golden.py

Models are read from the original pickles, not from the converted .npz files, so the golden outputs come from the
legacy code path. Regenerating overwrites tests/data and tests/golden.
"""
import os
import shutil
import sys
import tempfile

import pandas

import pipeline
from rsabias.core import classifier

LEGACY_PANDAS = '0.24.2'


def use_pickled_tables():
    original = classifier.resolve_table_path

    def pickle_only(filepath):
        path = os.path.splitext(filepath)[0] + '.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    classifier.resolve_table_path = pickle_only
    return original


def main():
    if pandas.__version__ != LEGACY_PANDAS:
        sys.exit('Golden outputs must come from the legacy environment (pandas {}), found pandas {}.'
                 .format(LEGACY_PANDAS, pandas.__version__))
    use_pickled_tables()

    for path in (pipeline.DATA, pipeline.GOLDEN):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)

    pipeline.export_keys(pipeline.KEYS)
    with open(pipeline.BABY_GROUPS, 'w') as f:
        pipeline.json.dump(pipeline.baby_groups(), f, indent=4, sort_keys=True)

    with tempfile.TemporaryDirectory() as tmp:
        golden_classify = os.path.join(pipeline.GOLDEN, 'classify')
        os.makedirs(golden_classify)
        for key_path in pipeline.key_files():
            out = os.path.join(tmp, 'classify', pipeline.key_name(key_path))
            os.makedirs(out)
            report = pipeline.classify(key_path, out)
            shutil.copyfile(report, os.path.join(golden_classify, pipeline.key_name(key_path) + '.json'))

        dist_dir = os.path.join(tmp, 'dist')
        for rel in pipeline.distributions(dist_dir):
            target = os.path.join(pipeline.GOLDEN, 'distributions', rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(os.path.join(dist_dir, rel), target)

        build_dir = os.path.join(tmp, 'build')
        os.makedirs(build_dir)
        pipeline.build(dist_dir, build_dir)
        os.makedirs(os.path.join(pipeline.GOLDEN, 'tables'))
        for table in pipeline.TABLES:
            shutil.copyfile(os.path.join(build_dir, table + '.npz'),
                            os.path.join(pipeline.GOLDEN, 'tables', table + '.npz'))

        test_dir = os.path.join(tmp, 'test_set')
        eval_dir = os.path.join(tmp, 'evaluate')
        os.makedirs(eval_dir)
        pipeline.filter_test_set(test_dir, tmp)
        pipeline.evaluate(test_dir, build_dir, eval_dir)
        os.makedirs(os.path.join(pipeline.GOLDEN, 'evaluate'))
        for name in pipeline.EVALUATE_OUTPUTS:
            shutil.copyfile(os.path.join(eval_dir, name), os.path.join(pipeline.GOLDEN, 'evaluate', name))

    print('Golden outputs written to {}'.format(pipeline.GOLDEN))


if __name__ == '__main__':
    main()
