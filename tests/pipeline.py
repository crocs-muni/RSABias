"""
Pipeline steps shared by generate_golden.py (run in the legacy image) and the tests, so both execute exactly the
same calls. Everything runs on the baby dataset, small enough for a test suite.
"""
import json
import math
import os
import random
import re

import numpy as np

from rsabias.core import classifier, dataset, tasks

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
MODEL = os.path.join(REPO, 'rsabias', 'model')
TRANS = os.path.join(MODEL, 'transformations', '5p_5q_blum_mod_roca.json')
GROUPS = os.path.join(MODEL, 'groups', 'groups.json')
DISTRIBUTIONS = os.path.join(MODEL, 'distributions')
SAMPLE_KEY = os.path.join(MODEL, 'sample_private_key.pem')
BABY = os.path.join(REPO, 'datasets', 'baby_dataset')

DATA = os.path.join(TESTS, 'data')
KEYS = os.path.join(DATA, 'keys')
BABY_GROUPS = os.path.join(DATA, 'groups_baby.json')
GOLDEN = os.path.join(TESTS, 'golden')

FILTER_SEED = 0
TABLES = ['classification_table_complex', 'classification_table_naive']
EVALUATE_OUTPUTS = ['n_1_results.json', 'n_10_results.json', 'n_20_results.json', 'n_30_results.json',
                    'n_100_results.json', 'confusion_matrix.txt']


def key_files():
    """The sample key shipped with the model plus keys exported from the baby dataset."""
    names = sorted(f for f in os.listdir(KEYS) if f.endswith('.pem'))
    return [SAMPLE_KEY] + [os.path.join(KEYS, f) for f in names]


def key_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def classify(key_path, out_dir):
    tasks.classify_key(key_path, out_dir)
    return os.path.join(out_dir, 'key_classification.json')


def dist_files(root):
    """Paths of all dist.json files under root, relative to it."""
    return sorted(os.path.relpath(os.path.join(r, 'dist.json'), root)
                  for r, _, files in os.walk(root) if 'dist.json' in files)


def distributions(out_dir):
    tasks.distributions(BABY, out_dir, TRANS)
    return dist_files(out_dir)


def build(dist_dir, out_dir):
    """Builds both tables; method 'complex' fills the naive table as well."""
    tasks.build_classification_table(TRANS, BABY_GROUPS, dist_dir, out_dir, 'complex')


def filter_test_set(out_dir, work_dir):
    """filter_test_dataset shuffles with the global random state and works in ./tmp of the current directory."""
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        random.seed(FILTER_SEED)
        tasks.filter_test_dataset(BABY, out_dir, BABY_GROUPS)
    finally:
        os.chdir(cwd)


def evaluate(test_dir, table_dir, out_dir):
    tasks.evaluate(test_dir, out_dir, TRANS, BABY_GROUPS, 'complex', table_dir)


def baby_groups():
    """Groups from groups.json that contain a baby dataset source, i.e. the classes a model built on it can learn."""
    with open(GROUPS) as f:
        groups = json.load(f)
    sources = {ds.meta.get_full_name() for ds in dataset.DataSet.find(BABY)}
    return {g: members for g, members in groups.items() if sources & set(members)}


def export_keys(out_dir, per_source=2):
    """Writes the first keys of each baby dataset source as PEM, to be classified like user supplied keys."""
    from Crypto.PublicKey import RSA
    os.makedirs(out_dir, exist_ok=True)
    for ds in dataset.DataSet.find(BABY):
        name = re.sub(r'[^A-Za-z0-9.]+', '_', ds.meta.get_full_name())
        for i, k in zip(range(per_source), ds.iterator()):
            pem = RSA.construct((k['n'], k['e'], k['d'], k['p'], k['q'])).export_key('PEM')
            with open(os.path.join(out_dir, '{}_{}.pem'.format(name, i)), 'wb') as f:
                f.write(pem)


def partition(groups):
    """Group labels are arbitrary; compare clusterings as sets of member sets."""
    return {frozenset(members) for members in groups.values()}


def assert_json_close(actual, expected, rel=1e-12, path='$'):
    """Exact comparison of JSON structures, with a relative tolerance for floats."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), path
        if math.isnan(expected):
            assert math.isnan(actual), path
        else:
            assert math.isclose(actual, expected, rel_tol=rel, abs_tol=0.0), \
                '{}: {} != {}'.format(path, actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected), \
            '{}: keys {} != {}'.format(path, list(actual), list(expected))
        for k in expected:
            assert_json_close(actual[k], expected[k], rel, '{}.{}'.format(path, k))
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), '{}: length differs'.format(path)
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_json_close(a, e, rel, '{}[{}]'.format(path, i))
    else:
        assert actual == expected, '{}: {!r} != {!r}'.format(path, actual, expected)


def assert_tables_close(actual_path, expected_path, rel=1e-12):
    """Golden tables come from the legacy code and carry its swapped feature names."""
    actual = classifier.fix_legacy_feature_names(classifier.load_classification_table(actual_path))
    expected = classifier.fix_legacy_feature_names(classifier.load_classification_table(expected_path))
    assert actual.dims == expected.dims
    for d in expected.dims:
        assert np.array_equal(actual.coords[d].values, expected.coords[d].values), d
    np.testing.assert_allclose(actual.values, expected.values, rtol=rel, atol=0)


def load_json(path):
    with open(path) as f:
        return json.load(f)
