import itertools
import os
import matplotlib as mpl
# mpl.use('Agg')  # back-end to run without X server
import matplotlib.pyplot as plt
import math
import sympy.ntheory as nt

import rsabias.core.features as features


class State:

    def __init__(self, fig, ax, cols, rows):
        self.fig = fig
        self.ax = ax
        self.cols = cols
        self.rows = rows


def ticks(values, labels, max_ticks=16):
    if len(values) <= max_ticks:
        return values, labels, list(range(len(values)))
    tick_ind = list(range(0, len(labels), len(labels) // max_ticks)) \
               + [len(labels) - 1]
    tick_val = [values[i] for i in tick_ind]
    tick_lab = [labels[i] for i in tick_ind]
    return tick_val, tick_lab, tick_ind


def normalize(counts, percentage=True):
    mul = 100 if percentage else 1
    sum_ = sum(counts)
    if sum_ == 0:
        return counts
    return [mul * c / sum_ for c in counts]


def figure(fig_ax=None, subplots=1, i=None, width=5, height=5):
    if fig_ax is not None:
        if i is None:
            return fig_ax
        cols = fig_ax.cols
        rows = fig_ax.rows
        ax = fig_ax.ax[i // cols][i % cols]
        ax.axis('on')
        return State(fig_ax.fig, ax, cols, rows)
    cols, rows = __cols_rows(subplots)
    fig, ax = plt.subplots(figsize=(width * cols, height * rows),
                           nrows=rows, ncols=cols)
    if rows > 1:
        for a in ax:
            if cols > 1:
                for b in a:
                    b.axis('off')
            else:
                a.axis('off')
    return State(fig, ax, cols, rows)


def add_ticks(values, labels, ax, x_axis=True, heat_map=False):
    tick_val, tick_lab, tick_ind = ticks(values, labels)
    if heat_map:
        tick_val = tick_ind
    if x_axis:
        ax.set_xticks(tick_val, minor=False)
        ax.set_xticklabels(tick_lab)
        long_x_labels = max([len(t) for t in tick_lab]) > 2
        if long_x_labels:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    else:
        ax.set_yticks(tick_val, minor=False)
        ax.set_yticklabels(tick_lab)


def plot_1d(dist, vlc, fig_ax=None, i=None):
    if vlc is None or dist is None:
        return None
    fig_ax = figure(fig_ax, i=i)
    values, labels, counts = vlc
    fig_ax.ax.bar(values, normalize(counts), width=1)
    add_ticks(values, labels, fig_ax.ax)
    plt.ylabel('Probability [%]')
    plt.xlabel(dist.feature_name)
    plt.title(dist.description)
    return fig_ax


def plot_overlay(dist, vlc_overlay, fig_ax=None, i=None):
    if vlc_overlay is None or dist is None:
        return None
    fig_ax = figure(fig_ax, i=i)
    values_list, counts_list, values, labels = vlc_overlay
    legend = []
    for v, c, d in zip(values_list, counts_list, dist.descriptions):
        p, = fig_ax.ax.plot(v, normalize(c), linewidth=3, alpha=0.4,
                            marker='.', label=d)
        legend.append(p)
    add_ticks(values, labels, fig_ax.ax)
    plt.ylabel('Probability [%]')
    plt.xlabel(dist.feature_name)
    plt.title(dist.description)
    fig_ax.ax.legend(handles=legend, loc=1)
    return fig_ax


def plot_2d(dist, heat_map, fig_ax=None, i=None):
    if heat_map is None or dist is None:
        return None
    fig_ax = figure(fig_ax, i=i)
    heat_map, vs, ls, cs = heat_map
    # TODO matplotlib.pyplot.pcolormesh
    cm = plt.get_cmap('jet')
    cm.set_bad('white')
    im = fig_ax.ax.imshow(heat_map, cmap=cm, origin='lower')
    cbar = fig_ax.ax.figure.colorbar(im, ax=fig_ax.ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    add_ticks(vs[0], ls[0], fig_ax.ax, True, True)
    add_ticks(vs[1], ls[1], fig_ax.ax, False, True)
    plt.xlabel(dist.descriptions[0])
    plt.ylabel(dist.descriptions[1])
    plt.title(dist.description)
    return fig_ax


# TODO smallest partitions
def __cols_rows(figures):
    if figures == 1:
        return 1, 1
    factors = nt.factorint(figures)
    expand = [[v]*count for v, count in factors.items()]
    expand = [i for sub in expand for i in sub]
    cols = 1
    rows = 1
    for f in reversed(expand):
        if rows < cols:
            rows *= f
        else:
            cols *= f
    if cols / rows > 2:
        cols = math.ceil(math.sqrt(figures))
        rows = math.ceil(figures / cols)
    return cols, rows


def save(fig_ax, save_path, name, file_format='svg'):
    if fig_ax is None:
        return None
    fig_ax.fig.tight_layout()
    plt.savefig(os.path.join(save_path, '{}.{}'.format(name, file_format)),
                dpi=600)
    plt.close(fig_ax.fig)


def plot(dist, save_path):
    if isinstance(dist, features.MultiFeatureDistribution):
        fig_ax = plot_overlay(dist, dist.vlc_overlay())
        save(fig_ax, save_path, dist.description)
        if fig_ax is None:
            for d in dist.counts:
                plot(d, save_path)
    elif isinstance(dist, features.MultiDimDistribution):
        # fig_ax = plot_1d(dist, dist.vlc())
        # save(fig_ax, save_path, dist.name)
        fig_ax = plot_overlay(dist, dist.vlc_overlay())
        save(fig_ax, save_path, '{}_overlay'.format(dist.description))
        fig_ax = plot_2d(dist, dist.heat_map())
        save(fig_ax, save_path, '{}_heat_map'.format(dist.description))

        dimensions = dist.dimensions
        for dim in range(2, dimensions):
            combinations = list(itertools.combinations(range(dimensions), dim))
            subplots = len(combinations)
            subspaces = [dist.subspace(list(c)) for c in combinations]
            # fig_ax_1d = figure(subplots=subplots)
            # for s, i in zip(subspaces, range(subplots)):
            #     plot_1d(s, s.vlc(), fig_ax_1d, i)
            # save(fig_ax_1d, save_path, '{}_{}'.format(dist.description, dim))
            # fig_ax_ol = figure(subplots=subplots)
            # for s, i in zip(subspaces, range(subplots)):
            #     plot_overlay(s, s.vlc_overlay(), fig_ax_ol, i)
            # save(fig_ax_ol, save_path, '{}_{}_overlay'.format(dist.description, dim))
            fig_ax_2d = figure(subplots=subplots)
            for s, i in zip(subspaces, range(subplots)):
                plot_2d(s, s.heat_map(), fig_ax_2d, i)
            save(fig_ax_2d, save_path, '{}_{}_heat_map'.format(dist.description, dim))
    elif isinstance(dist, features.Distribution):
        fig_ax = plot_1d(dist, dist.vlc())
        save(fig_ax, save_path, dist.description)
