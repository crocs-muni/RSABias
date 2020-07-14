import matplotlib.pyplot as plt
import seaborn as sn
import json
import ast
import os
import numpy as np
import xarray as xr
import pickle as pkl
import itertools
from timeit import default_timer as timer
from json import JSONEncoder

import rsabias.core.dataset as dataset
import rsabias.core.features as features


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Acc:
    """
    Simple dataclass to store accuracy values. We keep track of correct and wrong classification attempts.
    """
    def __init__(self, correct, wrong):
        self.correct = correct
        self.wrong = wrong


def save_object_pickle(obj, filepath):
    """
    Save obj into filepath as pickle object
    """
    with open(filepath, 'wb') as pickle_file:
        pkl.dump(obj, pickle_file)


def load_object_pickle(filepath):
    """
    Load obj from filepath as pickle object
    """
    with open(filepath, 'rb') as pickle_file:
        obj = pkl.load(pickle_file)
    return obj


def trans_from_path(trans_path):
    """
    Returns transformations as json object from path with the transformations.
    """
    with open(trans_path, mode='r') as f:
        trans = features.Parser.parse_dict(json.load(f))
    return trans


def distributions_from_files(data_sets_path, trans):
    """
    Function copied from rsabias.tasks to avoid circular dependencice
    """
    meta_filename = 'meta.json'
    dist_filename = 'dist.json'
    data_sets = dataset.DataSet.find(data_sets_path, meta_filename)
    metas = []
    dists = []
    for ds in data_sets:
        meta_path = os.path.join(ds.path, meta_filename)
        dist_path = os.path.join(ds.path, dist_filename)
        with open(meta_path, 'r') as meta:
            meta = dataset.DataSetMeta.import_dict(json.load(meta))
        with open(dist_path, 'r') as dist:
            distribution = features.Distribution.import_dict_json_safe(
                json.load(dist), trans)
        metas.append(meta)
        dists.append(distribution)
    return metas, dists


class ProbaTableException(Exception):
    pass


class Evaluator:
    """
    Use for evaluating classification success rate over complete test dataset with multiple parameters. Must be supplied
    with filled proba table object.
    """
    def __init__(self, test_data_sets_path, trans_path, groups_json_path, proba_table, method, output_path, top_n_conf=[1], n_keys_conf=[1]):
        self.test_data_sets_path = test_data_sets_path  # absolute path to folder with test data set
        self.trans = trans_from_path(trans_path)  # absolute path to transformations to compute on test dataset
        self.groups_json_path = groups_json_path  # absolute path to the file with group assignments
        self.proba_table = proba_table  # probability table to use for classification

        self.method = method  # either 'naive' or 'complex', which method to use for classification

        self.metas = []  # we will store meta files of test set here
        self.groups = []  # we will store assignment of groups to the meta files above (so we can zip(metas, groups))

        self.accuracies = {}  # We will store accuracies here (If provided with labels)
        self.frenquencies = {}  # We will store frequencies here (If not provided with labels)

        self.json_results = None # We will store a json object with results here

        self.top_n_conf = top_n_conf  # list of viable configurations for top_match parameter [1,2,3] etc..
        self.n_keys_conf = n_keys_conf  # list of viable configurations for batch n_keys configuration [1, 2, 3] etc...

        self.output_path = output_path  # we will store accuracies in json here

        # Dumb branching, unstable. This branch serves dataset classification
        if self.test_data_sets_path is not None:
            self.data_sets = dataset.DataSet.find(self.test_data_sets_path, 'meta.json')
            self.metas = [ds.meta for ds in self.data_sets]
            self.groups = [m.details.group for m in self.metas]
        # The branch below serves single key classification, when no dataset is provided
        else:
            self.data_sets = None
            self.metas = None
            self.groups = None

        self.classifier = Classifier(self.proba_table, self.method)

        self.confusion_matrix = None

        self.batch_frequencies = {}
        self.batch_cardinalities = {}
        self.batch_y_pred = {}
        self.batch_group_orderings = {}

    @staticmethod
    def find_group_in_dict(groups_dict, meta_name):
        for key, sources in groups_dict.items():
            if meta_name in sources:
                return key
        return None

    def init_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.proba_table.n_groups, self.proba_table.n_groups), dtype=np.float32)

    def dump_confusion_matrix(self, dirpath, title='Confusion matrix', normalize=True):
        # TODO: We have problems with normalizing when encountering zero records...
        """
        Plots a confusion matrix into the file
        :param cm: the sklearn object with confusion matrix
        :param dirpath: The directory to store the files into
        :param title: The main label of the plot
        :param normalize: Whether to print absolute numbers or to normalize the matrix to 0 - 100% scale
        :return: Nothing
        """
        filepath_png = os.path.join(dirpath, 'confusion_matrix.png')
        filepath_state = os.path.join(dirpath, 'confusion_matrix.pkl')
        filepath_text = os.path.join(dirpath, 'confusion_matrix.txt')

        accuracy = np.trace(self.confusion_matrix) / float(np.sum(self.confusion_matrix))
        misclass = 1 - accuracy

        if normalize is True:
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.1%'
        else:
            fmt = 'g'

        plt.figure(figsize=(40, 40))
        col_map = plt.get_cmap('Greens')

        hm = sn.heatmap(self.confusion_matrix, annot=True, cmap=col_map, fmt=fmt)
        hm.set_xticklabels(self.proba_table.groups_unique_sorted, rotation=45)
        hm.set_yticklabels(self.proba_table.groups_unique_sorted, rotation=45)

        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label\n\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

        plt.tight_layout()
        plt.savefig(filepath_png, bbox_inches='tight')

        with open(filepath_state, 'wb') as handle:
            pkl.dump(self.confusion_matrix, handle)

        with open(filepath_text, 'w') as handle:
            group_accuracies = list(self.confusion_matrix)
            for i, g in enumerate(group_accuracies):
                header_printed = False
                for j, acc in enumerate(g):
                    if acc > 0.05 and i != j:
                        if not header_printed:
                            handle.write(f'Group [{self.proba_table.groups_unique_sorted[i]}] is often missclassified as:\n')
                            header_printed = True
                        handle.write(f'\tgroup g[{self.proba_table.groups_unique_sorted[j]}] in {round(100 * acc, 1)}% cases\n')


    def evaluate_dataset(self, dset, y_true, n_keys):
        """
        Evaluates single dataset. It takes n_keys as a single key and computes all top_n match options specified in
        evaluator settings. It recovers values for respective groups from database. Then, we zip this array with group
        list and reorder it in descending order. If we find the result in top_n items, we increment successful attempts.
        We increment failed attempts otherwise.
        """

        y_pred = np.ones(self.proba_table.n_groups)
        keys_in_batch_classified = 0

        for key in dset.iterator():
            y_pred *= np.array(self.classifier.classify(key))
            keys_in_batch_classified += 1

            if keys_in_batch_classified == n_keys:
                group_ordering = self.proba_table.groups_unique_sorted
                y_pred, group_ordering = (list(t) for t in zip(*sorted(zip(list(y_pred), group_ordering), reverse=True)))

                for n in self.top_n_conf:
                    if y_true in group_ordering[:n]:
                        self.accuracies[n][y_true].correct += 1
                    else:
                        self.accuracies[n][y_true].wrong += 1

                    if n_keys == 1 and n == 1:
                        self.confusion_matrix[self.proba_table.groups_unique_sorted_hash_table[y_true], self.proba_table.groups_unique_sorted_hash_table[group_ordering[0]]] += 1

                y_pred = np.ones(self.proba_table.n_groups)
                keys_in_batch_classified = 0

    def evaluate_datasets(self, n_keys):
        """
        Evaluates classification for all datasets. n_keys specifies how many keys to classify at once (assuming identical origin).
        """
        self.accuracies = {x: {key: Acc(0, 0) for key in self.proba_table.groups_unique_sorted} for x in self.top_n_conf}

        for m in self.metas:
            print(f'{m.get_full_name()}: {m.count_records()}')

        for ds, g in zip(self.data_sets, self.groups):
            self.evaluate_dataset(ds, g, n_keys)

        self.json_results = {'n_keys': n_keys, 'accuracies': self.accuracies}
        filepath = os.path.join(self.output_path, 'n_' + str(n_keys) + '_results.json')
        with open(filepath, 'w') as file:
            json.dump(self.json_results, file, indent=4, cls=MyEncoder)

    def run_all_configurations(self):
        """
        This is a method to call when running evaluator.
        """
        self.init_confusion_matrix()
        for n in self.n_keys_conf:
            self.evaluate_datasets(n)
            if n == 1:
                self.dump_confusion_matrix(self.output_path)

    def classify_datasets(self):
        self.frenquencies = {key: 0 for key in self.proba_table.groups_unique_sorted}
        for ds in self.data_sets:
            self.classify_dataset(ds)

        self.json_results = {'frenquencies': self.frenquencies}
        filepath = os.path.join(self.output_path, 'frenquency_results.json')
        with open(filepath, 'w') as file:
            json.dump(self.json_results, file, indent=4, cls=MyEncoder)

    def classify_dataset(self, ds):
        for key in ds.iterator():
            y_pred = np.array(self.classifier.classify(key))
            group_ordering = self.proba_table.groups_unique_sorted
            y_pred, group_ordering = (list(t) for t in zip(*sorted(zip(list(y_pred), group_ordering), reverse=True)))
            best_guess = group_ordering[0]

            self.frenquencies[best_guess] += 1

    def init_batch_dictionary(self):
        # fill-in path to datasets/batch_gcd/rapid7/other/08230494f25e7f.json
        lines = [line.rstrip('\n') for line in open('see/comment/above', 'r')]
        for l in lines:
            json_data = json.loads(l)
            self.batch_frequencies[json_data['batch']] = np.ones(self.proba_table.n_groups)
            self.batch_cardinalities[json_data['batch']] = 0
            self.batch_y_pred[json_data['batch']] = [0] * self.proba_table.n_groups
            self.batch_group_orderings[json_data['batch']] = [None] * self.proba_table.n_groups

    def softmax(self, array):
        theta = 1.0  # determinism parameter
        ps = np.exp(array * theta)
        ps /= np.sum(ps)
        return ps

    def classify_batch_gcd(self):
        self.frenquencies = {key: 0 for key in self.proba_table.groups_unique_sorted}
        self.init_batch_dictionary()

        for key in self.data_sets[0].iterator():
            y_pred = np.array(self.classifier.classify(key))

            if np.count_nonzero(y_pred) == 0:
                print('Unseen key')

            self.batch_frequencies[key['batch']] *= y_pred
            self.batch_cardinalities[key['batch']] += 1
            """
            self.batch_frequencies[key['batch']] = self.softmax(y_pred) * self.softmax(self.batch_frequencies[key['batch']])
            self.batch_cardinalities[key['batch']] += 1
            """

            s = sorted(self.batch_frequencies[key['batch']], reverse=True)
            if 0 < s[0] < float('1.0e-20'):
                self.batch_frequencies[key['batch']] = [x * float('1.0e+20') for x in self.batch_frequencies[key['batch']]]

        self.json_results = {key: {'freq': val, 'batches': []} for key, val in self.frenquencies.items()}

        for batch_id, proba_vector in self.batch_frequencies.items():
            group_ordering = self.proba_table.groups_unique_sorted
            y_pred, group_ordering = (list(t) for t in zip(*sorted(zip(list(proba_vector), group_ordering), reverse=True)))
            best_guess = group_ordering[0]
            self.frenquencies[best_guess] += 1
            self.batch_group_orderings[batch_id] = group_ordering
            self.batch_y_pred[batch_id] = y_pred

            batch_data = {'batch id': batch_id, 'batch size': self.batch_cardinalities[batch_id], 'group ordering': group_ordering, 'likelihood vector': y_pred}
            self.json_results[best_guess]['batches'].append(batch_data)

        for group in self.json_results:
            self.json_results[group]['batches'].sort(key= lambda i: i['batch size'], reverse=True) # Should sor the list
            self.json_results[group]['freq'] = self.frenquencies[group]

        filepath = os.path.join(self.output_path, 'frenquency_results.json')
        with open(filepath, 'w') as file:
            json.dump(self.json_results, file, indent=4, cls=MyEncoder)


class Classifier:
    """
    Very simple class used to draw results from probability table. For given key computes the features and returns
    the guess of our classifier. Can be called either in 'naive' or 'complex' mode.
    """
    def __init__(self, proba_table, method):
        self.proba_table = proba_table
        if method == 'naive':
            self.table_to_look_at = self.proba_table.classification_table_naive
        elif method == 'complex':
            self.table_to_look_at = self.proba_table.classification_table_complex

    def classify(self, key):
        features = self.proba_table.trans.apply(key)
        dct = {key: val for key, val in zip(self.proba_table.dims[1:], features)}
        data = self.table_to_look_at.loc[dct].data.flatten()[0]
        return ast.literal_eval(data)


class ProbaTable:
    """
    The class used to store objects that hold all probability values.
    """
    def __init__(self, trans_path, groups_json_path, method):
        self.data_set_path = None  # path to the training dataset to construct the table from
        self.trans = trans_from_path(trans_path)  # path to transformations applied to this dataset
        self.groups_json_path = groups_json_path  # path to the grouping assignment

        self.trans_ranges = []  # range of values per each transformation (or feature if you will)
        self.groups = []  # will be used to match group for each dataset

        self.total_records_per_group = {}  # used to compute relative probabilities out of absolute frequencies. Holds total number of keys per group in dictionary, group is key.
        self.metas = None  # Will be used to hold meta datasets so that we can zip(metas, groups)
        self.dist = None   # distributions used to fill the tables, again aligned with metas and groups list

        # Data structures for Probability Tables
        self.da_complex = None  # DataArray to hold the whole table for Complex Bayes.
        self.da_naive = [] # Same but we hold a list of those (simplified arrays)

        self.classification_table_complex = None  # this is the table based on which we classify complex bayes
        self.classification_table_naive = None  # This is the table based on which we classify naive bayes

        self.n_groups = 0  # how many unique groups we have
        self.feature_names = []  # what are the names of features
        self.dims = []  # 'groups' + feature_names, used as labels for axes in data tables
        self.coords = [] # list of ranges of groups and features, used as boundaries for data tables
        self.trans_ranges = []  # ranges of all features

        self.groups_unique_sorted = []  # alphabetically sorted groups, duplicities deleted
        self.groups_unique_sorted_hash_table = {}

        self.method = method

    def init_common_structures(self):
        """
        Initialized structures that will be needed for whatever we will compute with the table
        :return:
        """
        self.groups_unique_sorted = self.get_list_of_unique_groups(self.groups_json_path)
        self.groups_unique_sorted_hash_table = {k: self.groups_unique_sorted.index(k) for k in self.groups_unique_sorted}
        self.n_groups = len(self.groups_unique_sorted)

        self.trans_ranges = self.get_trans_ranges()
        feature_coords = [self.get_coord_from_trans_range(r) for r in self.trans_ranges]
        self.coords = [self.groups_unique_sorted] + feature_coords

        self.feature_names = self.get_feature_names()
        self.dims = ['group'] + self.feature_names

    def init_tables_precomputation(self, data_set_path):
        """
        Initializes empty data arrays for selected table, either for 'naive' or 'complex'
        """
        self.data_set_path = data_set_path

        if not self.groups:
            self.get_meta_group_pairing()
        if not self.total_records_per_group:
            self.compute_total_group_records()

        total = 0
        for key, val in self.total_records_per_group.items():
            total += val
        print(f'Total: {total}')
        print(self.total_records_per_group)

        self.init_data_array_naive()
        self.init_data_array_complex()
        self.init_data_array_classification_tables()

    @staticmethod
    def get_list_of_unique_groups(groups_json_path):
        with open(groups_json_path) as json_groups:
            grps = json.load(json_groups)
        lst_of_groups = sorted(np.unique([key for key in grps.keys()]))
        return lst_of_groups

    @staticmethod
    def find_group_in_dict(groups_dict, meta_name):
        for key, sources in groups_dict.items():
            if meta_name in sources:
                return key
        return None

    def compute_total_group_records(self):
        self.total_records_per_group = {g: 0 for g in self.groups_unique_sorted}
        for m, g in zip(self.metas, self.groups):
            self.total_records_per_group[g] += m.count_records()

    def get_meta_group_pairing(self):
        """
        Fills self.groups in such way that both lists self.metas and self.groups are aligned. So that in self.grous[0]
        is the group of the meta dataset self.metas[0].
        """
        self.metas, self.dist = distributions_from_files(self.data_set_path, self.trans)

        with open(self.groups_json_path) as json_groups:
            grps = json.load(json_groups)

        for m in self.metas:
            self.groups.append(self.find_group_in_dict(grps, m.get_full_name()))

    @staticmethod
    def get_trans_ranges():
        # TODO: Not going to lie, the ranges of the features must be specified manually.

        # return [arr, [False, True], [1, 5, 251, 17863], [False, True]] # This is for Cross method
        return [range(0, 32, 1), range(0, 32, 1), [False, True], [1, 5, 251, 17863], [False, True]] # This is for Naive and Complex method

    @staticmethod
    def get_coord_from_trans_range(r):
        """
        Simply translates range object into list object if needed.
        :param r:
        :return:
        """
        if isinstance(r, range):
            return list(r)
        if isinstance(r, list):
            return r
        if isinstance(r, np.ndarray):
            return r
        return None

    @staticmethod
    def get_feature_names():
        # TODO: Not going to lie, the names of the features must be specified manually in the code.
        # return ['msb5', 'modularFingerprint', 'lsb', 'roca'] # This is for Cross feature method
        return ['msb5p', 'msb5q', 'modularFingerprint', 'lsb', 'roca']  # This is for complex and naive method

    def load_classification_tables(self, filepath, method):
        if method == 'naive':
            self.classification_table_naive = load_object_pickle(filepath)
        else:
            self.classification_table_complex = load_object_pickle(filepath)

    def build_classification_table(self, data_set_path, filepath_to_store_naive_table=None, filepath_to_store_complex_table=None):
        print('Initializing tables precomputation')
        self.init_tables_precomputation(data_set_path)

        if self.method == 'complex':
            print('Filling complex frequency table')
            self.fill_frequency_table_complex()

        print('Filling naive frequency table')
        if self.method == 'complex':
            self.fill_frequency_table_naive()
        if self.method == 'naive':
            self.fill_frequency_table_naive_from_dist()

        if self.method == 'complex':
            print('Filling complex classification table')
            self.fill_classification_table_complex(filepath_to_store_complex_table)

        print('Filling naive classification table')
        self.fill_classification_table_naive(filepath_to_store_naive_table)

    def init_data_array_complex(self):
        """
        Prepares empty array for distribution data.
        """
        shape = [len(x) for x in self.coords] # N+1 dimensional array, where n is number of features (+1 is group)
        empty_data = np.zeros(shape, dtype='int64')
        self.da_complex = xr.DataArray(empty_data, coords=self.coords, dims=self.dims, name='Complex Bayes Classification table')

    def init_data_array_naive(self):
        """
        Prepares empty array for distribution data. (marginalizes complex table)
        """
        for i in range(1, len(self.coords)): # we have own table for each feature
            shape = [len(self.coords[0]), len(self.coords[i])] # with two dimensions. group X feature
            coords = [self.coords[0], self.coords[i]]
            dims = [self.dims[0], self.dims[i]]
            empty_data = np.zeros(shape, dtype='int64')
            array_name = f'Naive Bayes Table for feature {self.dims[i]}'
            self.da_naive.append(xr.DataArray(empty_data, coords=coords, dims=dims, name=array_name))

    def init_data_array_classification_tables(self):
        """
        Prepares empty array for classification table. This is common call for both 'naive' and 'complex' and is constructed
        at once.
        """
        shape = [len(x) for x in self.coords[1:]]
        coords = self.coords[1:]
        dims = self.dims[1:]

        empty_data = np.empty(shape, dtype=object)
        self.classification_table_complex = xr.DataArray(empty_data, coords=coords, dims=dims,
                                                        name='Classification Table complex')
        self.classification_table_naive = xr.DataArray(empty_data, coords=coords, dims=dims,
                                                         name='Classification Table naive')

    def fill_frequency_table_complex(self):
        """
        Used to fill data table from distributions. Optionally, stores it into filepath as pickle object.
        """
        for group, dist in zip(self.groups, self.dist):
            for key, val in dist.counts.items():
                lst = [group] + list(key)
                self.da_complex.loc[dict(zip(self.dims, lst))] += np.uint32(val)
                #print(self.da_complex.loc[dict(zip(self.dims, lst))])

    def fill_frequency_table_naive(self):
        """
        Uses to fill data table from distributions. Optionally, stores it into filepath as pickle object.
        """
        for index, array in enumerate(self.da_naive):
            curr_feature_range = self.coords[index + 1]
            for group in self.groups_unique_sorted:
                for val in curr_feature_range:
                    retrieve = {'group': group, self.dims[index + 1]: val}
                    totals = sum(self.da_complex.loc[retrieve].data.flatten())
                    array.loc[group, val] = totals

    def fill_frequency_table_naive_from_dist(self):
        for group, dist in zip(self.groups, self.dist):
            for d, table in zip(dist.counts, self.da_naive):
                counts = d.counts
                for key, val in counts.items():
                    table.loc[group, key] += val

    def fill_classification_table_complex(self, filepath=None):
        """
        Fills complex classification table from data table and optionally stores it into filepath as pickle object.
        """
        start = timer()

        coords = self.coords[1:]

        for feature_combination in itertools.product(*coords):
            values = []

            dct = {key: val for key, val in zip(self.dims[1:], feature_combination)}
            subarray = self.da_complex.loc[dct]

            for group, records in self.total_records_per_group.items():
                values.append(int(subarray.loc[group]) / records)
            # values now hold an array of probabilities that is aligned with self.groups_unique_sorted
            self.classification_table_complex.loc[dct] = str(values)
            print(values)

        end = timer()
        print(f'Filling Classification table took {end - start} seconds.')

        if filepath is not None:
            save_object_pickle(self.classification_table_complex, filepath)

    def fill_classification_table_naive(self, filepath=None):
        """
        Fills naive classificaiton table and optionally stores it as a pickle object.
        """
        start = timer()

        coords = self.coords[1:]

        for feature_combination in itertools.product(*coords):
            values = [float(1) for _ in range(self.n_groups)]
            dct_to_store = {key: val for key, val in zip(self.dims[1:], feature_combination)}
            for i, f in enumerate(feature_combination):
                j = 0
                for group, records in self.total_records_per_group.items():
                    dct = {'group': group, self.dims[i+1]: f}
                    values[j] *= int(self.da_naive[i].loc[dct].data.flatten()[0]) / records
                    j += 1
            # values now hold an array of probabilities that is aligned with self.groups_unique_sorted
            self.classification_table_naive.loc[dct_to_store] = str(values)

        end = timer()
        print(f'Filling Classification table took {end - start} seconds.')
        if filepath is not None:
            save_object_pickle(self.classification_table_naive, filepath)
