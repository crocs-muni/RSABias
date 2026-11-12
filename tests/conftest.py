import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytest  # noqa: E402

import pipeline  # noqa: E402

# rsa_fingerprint.py and roca.py are standalone scripts in the repository root
sys.path.insert(0, pipeline.REPO)


@pytest.fixture(scope='session')
def dist_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('dist'))
    pipeline.distributions(out)
    return out


@pytest.fixture(scope='session')
def build_dir(tmp_path_factory, dist_dir):
    out = str(tmp_path_factory.mktemp('build'))
    pipeline.build(dist_dir, out)
    return out


@pytest.fixture(scope='session')
def eval_dir(tmp_path_factory, build_dir):
    work = str(tmp_path_factory.mktemp('filter_work'))
    test_dir = os.path.join(work, 'test_set')
    out = str(tmp_path_factory.mktemp('evaluate'))
    pipeline.filter_test_set(test_dir, work)
    pipeline.evaluate(test_dir, build_dir, out)
    return out
