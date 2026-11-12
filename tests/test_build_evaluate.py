import os

import pytest

import pipeline

pytestmark = pytest.mark.slow


@pytest.mark.parametrize('table', pipeline.TABLES)
def test_built_table_matches_golden(build_dir, table):
    pipeline.assert_tables_close(os.path.join(build_dir, table + '.npz'),
                                 os.path.join(pipeline.GOLDEN, 'tables', table + '.npz'))


@pytest.mark.parametrize('name', [n for n in pipeline.EVALUATE_OUTPUTS if n.endswith('.json')])
def test_evaluation_matches_golden(eval_dir, name):
    pipeline.assert_json_close(pipeline.load_json(os.path.join(eval_dir, name)),
                               pipeline.load_json(os.path.join(pipeline.GOLDEN, 'evaluate', name)))


def test_confusion_matrix_report_matches_golden(eval_dir):
    with open(os.path.join(eval_dir, 'confusion_matrix.txt')) as actual, \
            open(os.path.join(pipeline.GOLDEN, 'evaluate', 'confusion_matrix.txt')) as expected:
        assert actual.read() == expected.read()
