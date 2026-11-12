import os

import pipeline

GOLDEN_DIST = os.path.join(pipeline.GOLDEN, 'distributions')


def test_same_sources(dist_dir):
    assert pipeline.dist_files(dist_dir) == pipeline.dist_files(GOLDEN_DIST)


def test_distributions_match_golden(dist_dir):
    for rel in pipeline.dist_files(GOLDEN_DIST):
        expected = pipeline.load_json(os.path.join(GOLDEN_DIST, rel))
        pipeline.assert_json_close(pipeline.load_json(os.path.join(dist_dir, rel)), expected)
