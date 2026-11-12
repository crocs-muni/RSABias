import os

import pytest

import pipeline


@pytest.mark.parametrize('key_path', pipeline.key_files(), ids=pipeline.key_name)
def test_classify_matches_golden(key_path, tmp_path):
    report = pipeline.classify(key_path, str(tmp_path))
    expected = pipeline.load_json(os.path.join(pipeline.GOLDEN, 'classify', pipeline.key_name(key_path) + '.json'))
    pipeline.assert_json_close(pipeline.load_json(report), expected)
