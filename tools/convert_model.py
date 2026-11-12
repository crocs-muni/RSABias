"""
Converts a pickled classification table (legacy model format) into the pickle-free .npz format and verifies that
every cell round-trips exactly. Must run in the environment that wrote the pickle (legacy image, see README), as the pickle
references numpy/pandas/xarray internals of that time.

Usage: python tools/convert_model.py <classification_table.pkl> <groups.json>
"""
import ast
import os
import sys

import numpy as np

from rsabias.core import classifier


def main(pkl_path, groups_json_path):
    groups = classifier.ProbaTable.get_list_of_unique_groups(groups_json_path)
    legacy = classifier.fix_legacy_feature_names(classifier.load_object_pickle(pkl_path))
    npz_path = os.path.splitext(pkl_path)[0] + '.npz'
    classifier.save_classification_table(legacy, groups, npz_path)

    converted = classifier.load_classification_table(npz_path)
    if list(converted.coords['group'].values) != list(groups):
        sys.exit('Group order mismatch')
    for d in legacy.dims:
        if not np.array_equal(legacy.coords[d].values, converted.coords[d].values):
            sys.exit('Coordinates of {} differ'.format(d))
    expected = np.array([ast.literal_eval(c) for c in legacy.values.ravel()], dtype=np.float64)
    actual = converted.transpose(*(list(legacy.dims) + ['group'])).values.reshape(expected.shape)
    if not np.array_equal(expected, actual):
        sys.exit('Probabilities differ')
    print('Wrote {} ({} cells x {} groups), identical to {}'.format(npz_path, expected.shape[0], len(groups), pkl_path))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
