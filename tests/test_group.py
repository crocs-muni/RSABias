import json
import os

import pytest

import pipeline
from rsabias.core import tasks


@pytest.mark.slow
def test_clustering_reproduces_published_groups(tmp_path, capsys):
    tasks.cluster(pipeline.DISTRIBUTIONS, str(tmp_path), pipeline.TRANS)
    out = capsys.readouterr().out
    grouping = json.loads(out[out.index('{'):])
    assert pipeline.partition(grouping) == pipeline.partition(pipeline.load_json(pipeline.GROUPS))
    assert any(f.startswith('dendogram_') and f.endswith('.pdf') for f in os.listdir(str(tmp_path)))
