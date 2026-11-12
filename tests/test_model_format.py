import ast
import os

import numpy as np
import pytest

import pipeline
from rsabias.core import classifier

MODEL_NPZ = os.path.join(pipeline.MODEL, 'classification_table_complex.npz')
MODEL_PKL = os.path.join(pipeline.MODEL, 'classification_table_complex.pkl')


def load_legacy_pickle():
    """The pickle only loads with the dependency versions that wrote it."""
    try:
        return classifier.load_object_pickle(MODEL_PKL)
    except Exception as e:
        pytest.skip('legacy pickle not loadable in this environment: {}'.format(e))


def test_resolve_prefers_npz():
    assert classifier.resolve_table_path(MODEL_PKL) == MODEL_NPZ


def test_model_groups_match_groups_file():
    table = classifier.load_classification_table(MODEL_NPZ)
    groups = classifier.ProbaTable.get_list_of_unique_groups(pipeline.GROUPS)
    assert list(table.coords['group'].values) == list(groups)
    assert table.shape == (32, 32, 2, 4, 2, len(groups))


def test_npz_identical_to_legacy_pickle():
    legacy = classifier.fix_legacy_feature_names(load_legacy_pickle())
    converted = classifier.load_classification_table(MODEL_NPZ)
    for d in legacy.dims:
        assert np.array_equal(legacy.coords[d].values, converted.coords[d].values), d
    expected = np.array([ast.literal_eval(c) for c in legacy.values.ravel()], dtype=np.float64)
    actual = converted.transpose(*(list(legacy.dims) + ['group'])).values.reshape(expected.shape)
    assert np.array_equal(actual, expected)


def test_model_feature_names():
    """'lsb' is the boolean Blum feature, 'modularFingerprint' the small factor bound, as computed by the features."""
    table = classifier.load_classification_table(MODEL_NPZ)
    assert list(table.dims) == classifier.ProbaTable.get_feature_names() + ['group']
    assert list(table.coords['lsb'].values) == [False, True]
    assert list(table.coords['modularFingerprint'].values) == [1, 5, 251, 17863]


def test_legacy_feature_names_fixed_on_load():
    """Golden tables were built by the legacy code, with 'lsb' and 'modularFingerprint' swapped."""
    golden = os.path.join(pipeline.GOLDEN, 'tables', 'classification_table_complex.npz')
    assert classifier.load_classification_table(golden).dims[2:4] == ('modularFingerprint', 'lsb')
    table = classifier.ProbaTable(pipeline.TRANS, pipeline.BABY_GROUPS, 'complex')
    table.init_common_structures()
    table.load_classification_tables(golden, 'complex')
    assert list(table.classification_table_complex.dims) == classifier.ProbaTable.get_feature_names() + ['group']
    assert list(table.classification_table_complex.coords['lsb'].values) == [False, True]


def test_classification_tables_do_not_share_data():
    """Filling the naive table during build must not overwrite the complex one."""
    table = classifier.ProbaTable(pipeline.TRANS, pipeline.GROUPS, 'complex')
    table.init_common_structures()
    table.init_data_array_classification_tables()
    cell = dict(zip(table.feature_names, (0, 0, False, 1, False)))
    table.classification_table_naive.loc[cell] = 'naive'
    assert table.classification_table_complex.loc[cell].item() is None


def test_save_load_round_trip(tmp_path):
    table = classifier.load_classification_table(MODEL_NPZ)
    path = str(tmp_path / 'table.npz')
    classifier.save_classification_table(table, list(table.coords['group'].values), path)
    pipeline.assert_tables_close(path, MODEL_NPZ, rel=0)


def test_mismatching_groups_rejected(tmp_path):
    groups_path = tmp_path / 'groups.json'
    groups_path.write_text('{"1": [], "2": []}')
    table = classifier.ProbaTable(pipeline.TRANS, str(groups_path), 'complex')
    table.init_common_structures()
    with pytest.raises(classifier.ProbaTableException):
        table.load_classification_tables(MODEL_PKL, 'complex')
