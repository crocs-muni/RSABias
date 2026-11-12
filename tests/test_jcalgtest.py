"""
Keys in JCAlgTest format (output of the JCAlgTest smart card tool) carry only e, p and q; the importer derives n and d.
The test keys are baby dataset keys written out in that format.
"""
import hashlib
import io
import json
import math
import os

import pytest

import pipeline
from rsabias.core import dataset, key, tasks

STARSIGN = os.path.join('Card', 'G&D', 'StarSign', '512')


def starsign_keys():
    data_set = next(ds for ds in dataset.DataSet.find(pipeline.BABY) if ds.meta.source_path() == STARSIGN)
    return data_set, list(data_set.iterator())


def tlv(tag, value):
    """Tag, two-byte length and the value as big-endian hex, as JCAlgTest prints them."""
    h = '{:x}'.format(value)
    h = '0' * (len(h) % 2) + h
    return '{}{:04x}{}'.format(tag, len(h) // 2, h)


def to_jcalgtest(k, status_word_after_p=False):
    priv = tlv('83', k['p']) + ('9000' if status_word_after_p else '') + tlv('84', k['q'])
    return 'PUBL: {}\nPRIV: {}\n# {}:{}\n'.format(tlv('81', k['e']), priv, k['id'], k['t'] / 1000)


@pytest.mark.parametrize('status_word_after_p', [False, True])
def test_import_alg_test(status_word_after_p):
    _, keys = starsign_keys()
    for k in keys:
        imported = key.Key.import_alg_test(io.StringIO(to_jcalgtest(k, status_word_after_p)))
        for part in ('id', 'n', 'e', 'p', 'q'):
            assert imported[part] == k[part], part
        assert imported['e'] * imported['d'] % math.lcm(k['p'] - 1, k['q'] - 1) == 1


def test_jcalgtest_dataset_matches_golden_distribution(tmp_path):
    """The features use p, q and n only, so the distribution must equal the one of the original JSON data set."""
    data_set, keys = starsign_keys()
    source_dir = tmp_path / 'source' / STARSIGN
    source_dir.mkdir(parents=True)
    content = ''.join(to_jcalgtest(k) for k in keys).encode()
    (source_dir / 'keys.txt').write_bytes(content)

    meta = data_set.meta.export_dict()
    meta['details'].update(format='jcalgtest', compressed=False)
    meta['files'] = [{'name': 'keys.txt', 'records': len(keys), 'sha256': hashlib.sha256(content).hexdigest()}]
    (source_dir / 'meta.json').write_text(json.dumps(meta))

    out = tmp_path / 'dist'
    tasks.distributions(str(tmp_path / 'source'), str(out), pipeline.TRANS)
    expected = pipeline.load_json(os.path.join(pipeline.GOLDEN, 'distributions', STARSIGN, 'dist.json'))
    pipeline.assert_json_close(pipeline.load_json(str(out / STARSIGN / 'dist.json')), expected)
