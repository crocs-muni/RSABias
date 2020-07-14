import pickle
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib import cm as cmplt

# This file represents a simple function to dump confusion matrix obtain by the evaluator of our classifier. Similar
# matrix was produced for the original paper.


def dump_confusion_matrix(filepath, cm_path, groups_unique_sorted):
    """
    A simple function to dump confusion matrix obtain by the evaluator of our classifier. Similar
    matrix was produced for the original paper.
    :param filepath: Path where the confusion matrix will be stored.
    :param cm_path: Path to the confusion matrix pickle object produced by the sklearn. This object is automatically
    created in the folder with the results by the "evaluate" task.
    :param groups_unique_sorted: The names of the groups that will be used as labels. They should be aligned with the
    sklearn object order of rows/columns.
    :return:
    """
    with open(cm_path, 'rb') as pkl_file:
        cm = pickle.load(pkl_file)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.0%'

    # Sorting procedure below. Convert to DataFrame, sort, convert to NumpyArray
    groups_ints = [int(x) for x in groups_unique_sorted]
    df_conf_matrix = pd.DataFrame(cm, columns=groups_ints)
    df_conf_matrix = df_conf_matrix[[x for x in range(1, len(groups_ints) + 1)]]
    df_conf_matrix['sorter'] = groups_ints
    df_conf_matrix = df_conf_matrix.sort_values(by='sorter')
    df_conf_matrix = df_conf_matrix.drop(['sorter'], axis=1)
    cm = df_conf_matrix.to_numpy()
    groups_ints, groups_unique_sorted = (list(t) for t in zip(*sorted(zip(groups_ints, groups_unique_sorted))))

    viridis = cmplt.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([155 / 256, 221 / 256, 155 / 256, 1])
    newcolors[:25, :] = pink

    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(255 / 256, 165 / 256, N)
    vals[:, 1] = np.linspace(255 / 256, 225 / 256, N)
    vals[:, 2] = np.linspace(255 / 256, 165 / 256, N)
    newcmp = ListedColormap(vals)

    plt.figure(figsize=(8, 6.9))
    col_map = plt.get_cmap(newcmp)

    hm = sn.heatmap(cm, annot=True, cmap=col_map, fmt=fmt)
    hm.set_xticklabels(groups_unique_sorted, rotation=45)
    hm.set_yticklabels(groups_unique_sorted, rotation=45)

    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    plt.title('Confusion matrix')
    plt.ylabel('True group', size=12)
    plt.xlabel('Group predicted by our model'.format((accuracy * 100), (misclass * 100)), size=12)

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', format='eps', dpi=300)
