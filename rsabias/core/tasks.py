"""
Common high level tasks to be used from e.g. main.
"""
import json
import os
import shutil
import random
import copy
import hashlib
import numpy as np
import pkg_resources


from rsabias.core import dataset, features, plot, table, classifier, visualizer, key


def classify_key(input_path, output_path):
    """
    Classifies a single key using a precomputed model with known precision/recall statistics.
    :param input_path: The path to the der/pem file with the private key. Must not be protected with password.
    :param output_path: Path where the classifier will output the full classification report in json.
    :return: Nothing
    """
    model_root_path = pkg_resources.resource_filename('rsabias.model', '')
    trans_path = os.path.join(model_root_path, 'transformations/5p_5q_blum_mod_roca.json')
    groups_json_path = os.path.join(model_root_path, 'groups/groups.json')
    prec_recall_path = os.path.join(model_root_path, 'precision_recall.json')
    method = 'complex'
    report_path = os.path.join(output_path, 'key_classification.json')
    table_path = os.path.join(model_root_path, 'classification_table_complex.pkl')

    with open(prec_recall_path, 'r') as json_handle:
        prec_recall = json.load(json_handle)

    with open(input_path, 'r') as handle:
        try:
            key_pem = handle.read()
        except Exception as e:
            print(f'Failed to read key-data from file, {e}.')
            return

    h = hashlib.new('sha256')
    h.update(str(key_pem).encode('utf-8'))

    try:
        key_to_classify = key.Key().import_standard_format(key_pem)
    except Exception as e:
        print(f'Failed to build a key object from the key, {e}.')
        return

    proba_table = classifier.ProbaTable(trans_path, groups_json_path, method)
    proba_table.init_common_structures()
    proba_table.load_classification_tables(table_path, method)

    model = classifier.Classifier(proba_table, method)
    y_pred = model.classify(key_to_classify)
    group_ordering = proba_table.groups_unique_sorted
    y_pred, group_ordering = (list(t) for t in zip(*sorted(zip(list(y_pred), group_ordering), reverse=True)))

    with open(groups_json_path, 'r') as json_handle:
        groups_data = json.load(json_handle)

    print(f'Your key with sha-256 digest: {h.hexdigest()} gets classified as:')
    print(f'\t- Classification group: {group_ordering[0]}')
    print(f'\t- Score: {y_pred[0]}')
    precision = prec_recall[group_ordering[0]]['Precision']
    recall = prec_recall[group_ordering[0]]['Recall']
    print(f'\t- If a key is marked as coming from Group {group_ordering[0]}, the model is in {100*precision:.2f}% cases right.')
    print(f'\t- The model correctly classifies {100*recall:.2f}% keys that are actually from Group {group_ordering[0]}.')
    print(f'\t- This group contains the following sources:')
    for src in groups_data[group_ordering[0]]:
        print(f'\t\t* {src}')
    print(f'\t- The full report of the key classification can be found at: {report_path}')

    report_dict = {}
    for score, group in zip(y_pred, group_ordering):
        group_dict = {'group_name': 'Group ' + group, 'score': score, 'Precision': prec_recall[group]['Precision'], 'Recall': prec_recall[group]['Recall'], 'sources': groups_data[group]}
        report_dict[group] = group_dict

    with open(report_path, 'w') as json_handle:
        json.dump(report_dict, json_handle, indent=4)


def convert_data_set(data_set, data_set_writer, transformation=None,
                     count_dist=False, prime_wise=False, remove_duplicities=False):
    """
    Convert data set to a different format.

    If transformation is provided, compute features and add them to the key
    key that gets saved using a data_set_writer.

    If count_dist is True, also compute and return the distributions.

    :param data_set: source dataset.DataSet
    :param data_set_writer: target dataset.Writer
    :param transformation: optional features.Transformation
    :param count_dist: if True, compute the distributions
    :param prime_wise: iterate over primes rather than keys
    :param remove_duplicities: remove identical keys before processing
    :return: distribution for the top level transformation
             if count_dist is True, else None
    """
    feats = []
    seen_keys = set()
    for key in data_set.iterator(prime_wise=prime_wise):
        if remove_duplicities:
            digest = hashlib.sha256()
            digest.update(str(key).encode('utf-8'))
            d = digest.hexdigest()
            if d in seen_keys:
                continue
            seen_keys.add(d)
        if transformation:
            f = transformation.apply(key)
            if count_dist:
                feats.append(f)
        data_set_writer.write(key)
    if count_dist and transformation:
        return transformation.tally(feats)
    return None


def split_dataset(data_set, train_writer, test_writer, n_test_keys):
    """
    Splits a single dataset into test/train part. Method assumes each data_set to have enough keys to withdraw!
    :param data_set: The dataset to split
    :param train_writer: The writer to store train keys with
    :param test_writer: The writer to store test keys with
    :param n_test_keys: How many test keys to use
    :return: Nothing really...
    """
    dataset_n_keys = data_set.meta.count_records()

    keys_to_test_set = sorted(random.sample(range(dataset_n_keys), n_test_keys))

    for i, key in enumerate(data_set.iterator()):
        if keys_to_test_set and keys_to_test_set[0] == i:
            test_writer.write(key)
            keys_to_test_set.pop(0)
        else:
            train_writer.write(key)


def evaluate(test_sets_path, output_path, trans_path, groups_json_path, method, classification_table_dir, labels=True):
    """
    Evaluates classification accuracy on a test dataset.
    :param test_sets_path: directory with the test dataset to classify
    :param output_path: path to directory where the results should be stored
    :param trans_path: path with the transformations that shall be used on the keys (features)
    :param groups_json_path: path to the source - group mapping
    :param method: whether to use complex bayess or naive bayess
    :param classification_table_dir: path to the directory with saved classification tables
    :param labels: If true, we evaluate success rate. Else, we simply classify the whole dataset according to our best efforts
    :return: nothing
    """
    if method == 'naive':
        table_path = os.path.join(classification_table_dir, 'classification_table_naive.pkl')
    else:
        table_path = os.path.join(classification_table_dir, 'classification_table_complex.pkl')

    table = classifier.ProbaTable(trans_path, groups_json_path, method)
    table.init_common_structures()
    table.load_classification_tables(table_path, method)

    eval = classifier.Evaluator(test_sets_path, trans_path, groups_json_path, table, method, output_path, [1,2,3], [1, 10, 20, 30, 100])
    eval.run_all_configurations()


def visualize_model_performance(input_path, output_path):
    """
    This function visualizes table of the model performance. See ../classification_table_template/table_example.png for
    and example.
    :param input_path: The path with the json evaluated model.
    :param output_path: The path where the table tex file will be stored.
    :return: nothing
    """
    template_path = pkg_resources.resource_filename('rsabias.classification_table_template', '')
    viz = visualizer.ClassificationTableVisualizer(input_path, output_path, template_path)
    viz.construct_table()


def build_classification_table(trans_path, groups_json_path, data_set_path, output_path, method):
    """
    Task to build the classification tables from the dataset.
    :param trans_path: Path to the transformation/features.
    :param groups_json_path: Path to the groups that are result of a clustering task
    :param data_set_path: Path to the dataset to build the tables from
    :param output_path: This is where the tables will be stored at
    :param method: What model to use: complex, naive, cross
    :return:
    """
    np.set_printoptions(suppress=True)
    naive_table_path = os.path.join(output_path, 'classification_table_naive.pkl')
    complex_table_path = os.path.join(output_path, 'classification_table_complex.pkl')
    table_path = os.path.join(output_path, 'proba_table.pkl')
    table = classifier.ProbaTable(trans_path, groups_json_path, method)
    table.init_common_structures()
    table.build_classification_table(data_set_path, naive_table_path, complex_table_path)
    classifier.save_object_pickle(table, table_path)


def batch_gcd(input_path, output_path, transformation_path, groups_path, method, classtable_path):
    """
    Full pipeline of classifying the gcd-factorized dataset. First, tables are built from distributions, then, keys are classified.
    :param input_path: path to the gcd-factorized dataset
    :param output_path: path to the folder where results will be stored
    :param transformation_path: path to the transformations to-be-applied on the dataset (also called features)
    :param groups_path: path to the groups that are result of the clustering process
    :param method: what model to use for evaluation (complex, naive, cross)
    :param classtable_path: path to the folder with classification tables
    :return:
    """
    table_path = os.path.join(classtable_path, 'classification_table_complex.pkl')
    table = classifier.ProbaTable(transformation_path, groups_path, method)
    table.init_common_structures()
    table.load_classification_tables(table_path, method)
    eval = classifier.Evaluator(input_path, transformation_path, groups_path, table, method, output_path, [1], [1])
    eval.classify_batch_gcd()


def split(data_sets_path, output_path, meta_filename='meta.json', out_format='json', compress=True, n_test_keys=10000):
    """
    Splits the dataset into test/train part. If some source has not enough keys for splitting, it will simply be ignored.
    The function also discards all dataset for which private keys are not available
    and the name of the source printed to the standard output.
    :param data_sets_path: path to the original dataset to split
    :param output_path: Where to store both split parts
    :param meta_filename: What is the filename of metafiles in the original dataset
    :param out_format: Resulting format of the split datasets
    :param compress: Whether the split datasets should be compressed or not
    :param n_test_keys: How many test keys to include into the split dataset.
    :return:
    """
    if output_path[-1] != '/':
        split_dataset_path = output_path + '/' + 'split'
    else:
        split_dataset_path = output_path + 'split'

    success = False

    while not success:
        try:
            os.mkdir(split_dataset_path)
            success = True
        except FileExistsError:
            split_dataset_path += '+'
            success = False

    train_path = split_dataset_path + '/train'
    test_path = split_dataset_path + '/test'
    os.mkdir(train_path)
    os.mkdir(test_path)

    data_sets = dataset.DataSet.find(data_sets_path, meta_filename)
    metas = [ds.meta for ds in data_sets]

    to_remove = []
    for m in metas:
        if m.count_records() <= n_test_keys:
            print(f'Failed to add source due to insufficient number of keys: {m.details.name} {m.details.version}')
            to_remove.append(m)
        if m.details.public_only is True:
            print(f'Failed to add source due to having no private keys: {m.details.name} {m.details.version}')
            to_remove.append(m)
            
    metas = [m for m in metas if m not in to_remove]

    train_writers = [dataset.Writer.import_meta(train_path, meta, out_format, compress, header=None, separator=None) for meta in metas]
    test_writers = [dataset.Writer.import_meta(test_path, meta, out_format, compress, header=None, separator=None) for meta in metas]

    for ds, train_writer, test_writer in zip(data_sets, train_writers, test_writers):
        split_dataset(ds, train_writer, test_writer, n_test_keys)
        train_writer.close()
        test_writer.close()


# This function is a helper, not a separate task
def build_group_metas_dict(metas, groups):
    groups_sorted_unique = sorted(list(np.unique(groups)))
    group_meta_dict = {key: [] for key in groups_sorted_unique}

    for i, m in enumerate(metas):
        group_meta_dict[groups[i]].append(m)

    return group_meta_dict

# This function is a helper, not a separate task
def find_group_in_dict(groups_dict, meta_name):
    for key, sources in groups_dict.items():
        if meta_name in sources:
            return key
    return None

# This function is a helper, not a separate task
def get_group_list(metas, groups_json_path):
    groups = []
    with open(groups_json_path) as json_groups:
        grps = json.load(json_groups)

    for m in metas:
        groups.append(find_group_in_dict(grps, m.get_full_name()))
    return groups


def store_to_single_dir(data_sets_path, output_path, group, metas_list):
    """
    Creates new meta file for group and stores all sources into the same directory. At the same time, writes the records
    about the sources into the meta.json file.
    :param data_sets_path: root path to the datasets
    :param output_path: where to store the merged datasets.
    :param group: the name of the group (to look for in the dataset)
    :param metas_list: list of metas belonging to the group
    :return the meta file of the merged dataset
    """
    if not metas_list:
        return

    group_meta = copy.deepcopy(metas_list[0])
    group_meta.details.bitlen = 'mixed'
    group_meta.details.category = 'Group'
    group_meta.details.name = group
    group_meta.details.version = ''
    group_meta.files = []
    group_meta.details.group = group

    # make dir for new group
    group_path = os.path.join(output_path, group_meta.source_path())
    os.makedirs(group_path)

    for m in metas_list:
        for f in m.files:
            absolute_file_path = os.path.join(data_sets_path, m.source_path(), f.name)
            absolute_new_file_path = os.path.join(group_path, f.name)
            shutil.copyfile(absolute_file_path, absolute_new_file_path)
            group_meta.files.append(f)

    meta_out_path = os.path.join(group_path, 'meta.json')
    with open(meta_out_path, mode='w') as f:
        f.writelines(json.dumps(group_meta.export_dict(), sort_keys=True, indent=4))

    return group_meta


def shuffle_datasets(data_sets_path):
    """
    Shuffles all datasets, done in memory (not meant for huge datasets).
    :param data_sets_path: root path of the datasets to shuffle.
    :return:
    """
    data_sets = dataset.DataSet.find(data_sets_path, 'meta.json')
    metas = [ds.meta for ds in data_sets]

    for m in metas:
        for f in m.files:
            filepath = os.path.join(data_sets_path, m.source_path(), f.name)
            with open(filepath, 'r') as src:
                data = [(random.random(), line) for line in src]
                data.sort()
            new_filepath = os.path.join(data_sets_path, m.source_path(), 'new')
            with open(new_filepath, 'w') as file:
                for _, line in data:
                    file.write(line)
            rename_file_according_to_hash(new_filepath)


def rename_file_according_to_hash(filepath):
    """
    Renames file to its first 8 bytes of sha256 digest and modifies this record in the corresponding meta file.
    :param filepath: where both meta.json and the file are present
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    digest = sha256.hexdigest()
    old_path = filepath
    dir_path = os.path.dirname(old_path)
    meta_path = os.path.join(dir_path, 'meta.json')
    old_name = os.path.basename(old_path)
    new_name = old_name.replace(old_name.split('.')[0], digest[0:16]) + '.json'
    new_path = os.path.join(dir_path, new_name)
    os.rename(old_path, new_path)

    with open(meta_path, 'r') as json_file:
        data = json.load(json_file)
        old_data_filepath = os.path.join(dir_path, data['files'][0]['name'])
        os.remove(old_data_filepath)
        data['files'][0]['name'] = new_name
        data['files'][0]['sha256'] = digest

    with open(meta_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def filter_test_dataset(data_sets_path, output_path, groups_json_path, meta_filename='meta.json', out_format='json'):
    """
    This is the task to call for preparing test dataset. The function does several things:
        1.) It creates a meta dataset for all sources within one group
        2.) It merges all those sources into one file
        3.) It shuffles the file randomly
        4.) It recomputes hashes and stores those into the corresponding meta file (this is a bumpy ride, own functions used)
    :param data_sets_path:  path to the dataset to prepare
    :param output_path:  path where to store the final dataset
    :param groups_json_path: path to the group assignment
    :param meta_filename: 'meta.json'
    :param out_format: whether to store in json or not. We force decompressed format, as we need to shuffle (and not convert later on)
    """
    data_sets = dataset.DataSet.find(data_sets_path, meta_filename)
    metas = [ds.meta for ds in data_sets]
    groups = get_group_list(metas, groups_json_path)
    group_meta_dict = build_group_metas_dict(metas, groups)
    group_metas = []

    curr_dirr = os.getcwd()
    tmp_dir = os.path.join(curr_dirr, 'tmp')
    for group, metas_list in group_meta_dict.items():
        group_metas.append(store_to_single_dir(data_sets_path, tmp_dir, group, metas_list))

    group_writers = [dataset.Writer.import_meta(output_path, meta, out_format, False, header=None, separator=None) for meta in group_metas]
    group_datasets = dataset.DataSet.find(tmp_dir, meta_filename)

    for wr in group_writers:
        for ds in group_datasets:
            if wr.meta.details.group == ds.meta.details.group:
                for key in ds.iterator():
                    wr.write(key)
                wr.close()

    shutil.rmtree(tmp_dir)
    shuffle_datasets(output_path)


def convert(data_sets_path, output_path, trans_path=None,
            count_dist=True, distributions_only=False,
            out_format='json', compress=True, draw_plot=False,
            prime_wise=False, remove_duplicities=False):
    """
    Convert multiple data sets into a different format.

    :param data_sets_path: directory path for the input data sets
    :param output_path: directory path for the output
    :param trans_path: optional file path for the transformation file,
                       if provided, extra features will be computed
    :param count_dist: if True, distributions will be computed
    :param distributions_only: if True, only computes the distributions,
                               does not write keys to the output data set
    :param out_format: output format (e.g., "json", "csv")
    :param compress: compress the output using GZIP if True
    :param draw_plot: if True, distributions will be plotted
    :param prime_wise: iterate over primes rather than keys
    :param remove_duplicities: remove identical keys (inside individual data sets)
    :return:
    """
    trans = None
    new_bases = None
    if trans_path:
        with open(trans_path, mode='r') as f:
            trans = features.Parser.parse_dict(json.load(f))
        new_bases = trans.base_dict()
    meta_filename = 'meta.json'

    data_sets = dataset.DataSet.find(data_sets_path, meta_filename)
    metas = [ds.meta for ds in data_sets]
    writers = [dataset.Writer.import_meta(output_path, meta, out_format,
                                          compress, header=None,
                                          separator=None, new_bases=new_bases,
                                          skip_writing=distributions_only)
               for meta in metas]
    for ds, wr in zip(data_sets, writers):
        dist = convert_data_set(ds, wr, trans, count_dist=count_dist,
                                prime_wise=prime_wise,
                                remove_duplicities=remove_duplicities)
        if count_dist and dist:
            wr.save_distributions(dist)
            if draw_plot:
                if dist.plotable:
                    plot.plot(dist, wr.path)
        wr.close()  # TODO closing and exception using with


def distributions(data_sets_path, output_path, trans_path, draw_plot=False,
                  prime_wise=False):
    """
    Compute and save the distributions.

    :param data_sets_path: directory path for the input data sets
    :param output_path: directory path for the output
    :param trans_path: file path for the transformation file
    :param draw_plot: if True, distributions will be plotted
    :param prime_wise: iterate over primes rather than keys
    :return:
    """
    convert(data_sets_path, output_path, trans_path, count_dist=True,
            distributions_only=True, out_format=None, compress=None,
            draw_plot=draw_plot, prime_wise=prime_wise)


def plot_dist(data_sets_path, output_path, trans_path):
    """
    Plot precomputed distributions.

    :param data_sets_path: directory path for the precomputed distributions
    :param output_path: directory path for the output
    :param trans_path: file path for the transformation file
    :return:
    """
    meta_filename = 'meta.json'
    dist_filename = 'dist.json'
    # TODO transformations from distribution names
    with open(trans_path, mode='r') as f:
        trans = features.Parser.parse_dict(json.load(f))
    data_sets = dataset.DataSet.find(data_sets_path, meta_filename)
    for ds in data_sets:
        meta_path = os.path.join(ds.path, meta_filename)
        dist_path = os.path.join(ds.path, dist_filename)
        with open(dist_path, mode='r') as f:
            dist_dicts = json.load(f)
        out_path = os.path.join(output_path, ds.meta.source_path())
        created = False
        while not created:
            try:
                os.makedirs(out_path)
                created = True
            except FileExistsError:
                out_path += '+'
        df = features.Distribution.import_dict_json_safe(dist_dicts, trans)
        plot.plot(df, out_path)
        new_meta_path = os.path.join(out_path, meta_filename)
        new_dist_path = os.path.join(out_path, dist_filename)
        shutil.copy2(meta_path, new_meta_path)
        shutil.copy2(dist_path, new_dist_path)


def trans_from_path(trans_path):
    """
    Read a Transformation from a file on a file path.
    :param trans_path: file path for the transformation file
    :return: transformation
    """
    with open(trans_path, mode='r') as f:
        trans = features.Parser.parse_dict(json.load(f))
    return trans


def distributions_from_files(data_sets_path, trans):
    """
    Read distributions from files.
    :param data_sets_path: directory path for the precomputed distributions
    :param trans: transformation - features used to compute distributions
    :return: list of dataset.DataSetMeta and list of features.Distribution
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


def cluster(data_sets_path, output_path, trans_path):
    """
    Create the clustering (groups) from distributions of individual sources.
    Write down the results of this method. Currently, print on the standard output. In a new way, will be printed into
    the file. Move this method into table.py.

    :param data_sets_path: directory path for the precomputed distributions
    :param output_path: directory path for the output
    :param trans_path: transformation - features used to compute distributions
    :return:
    """
    trans = trans_from_path(trans_path)
    metas, dists = distributions_from_files(data_sets_path, trans)
    groups = table.Table.group(metas, dists, trans, out_path=output_path)
    labels = [' '.join(m.source()) for m in metas]
    grouping = dict()
    for l, g in zip(labels, groups):
        if g not in grouping:
            grouping[g] = []
        grouping[g].append(l)
    print(json.dumps(grouping, indent=4))


def marginal(data_sets_path, output_path, trans_path, subspaces):
    trans = trans_from_path(trans_path)
    metas, dists = distributions_from_files(data_sets_path, trans)
    if subspaces is None:
        subspaces = [[x] for x in range(0, len(trans.trans))]
    elif isinstance(subspaces, str):
        subspaces = json.loads(subspaces)
    elif not isinstance(subspaces, list):
        raise Exception('Subspaces must be list, str or None')
    table.Table.marginalize_partial(metas, dists, trans, subspaces, output_path)


def cluster_compare(data_sets_path, output_path, trans_path):
    """
    Compare clustering metrics.

    TODO: present experiment results or delete

    :param data_sets_path: directory path for the precomputed distributions
    :param output_path: directory path for the output
    :param trans_path: transformation - features used to compute distributions
    :return:
    """
    trans = trans_from_path(trans_path)
    metas, dists = distributions_from_files(data_sets_path, trans)
    labels = [' '.join(m.source()) for m in metas]
    norms = [1, 2, 3, 0.5, 0.25, 0.75]
    groupings = []
    # use different distance norms
    for norm in norms:
        groups = table.Table.group(metas, dists, trans, norm, output_path)
        groupings.append(groups)
    source_groups = sorted(zip(zip(*groupings), range(0, len(metas))))
    for g, s in source_groups:
        print(s, labels[s], g)

    # decide on threshold based on known division of groups (grouping hint)
    #   smallest distance for sources that should be different
    #   must be different for each feature :(
    # produce comparison table
    # produce dendrograms
    pass
