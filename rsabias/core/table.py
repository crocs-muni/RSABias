import itertools
import scipy.cluster.hierarchy as cluster
import matplotlib.pyplot as plt
import os.path
import rsabias.core.features as features
import rsabias.core.dataset as dataset


class TableException(Exception):
    pass


def show_dendrogram(linkage, labels, threshold):
    d = cluster.dendrogram(linkage, labels=labels,
                           above_threshold_color='gray',
                           orientation='left', color_threshold=threshold)
    # TODO color labels:
    # http://www.nxn.se/valent/extract-cluster-elements-by-color-in-python
    plt.axvline(x=threshold)
    # TODO collapse source versions deep under threshold


class Table:

    def __init__(self, top_transformation):
        self.trans = top_transformation

    def import_json(self):
        pass

    def export_json(self):
        pass

    @staticmethod
    def group(metas, dists, trans, norm=1, out_path=None, threshold_hint=None):
        """
        Add treshold as a parameter. If defined, save fig (matplotlib). If not defined, set default treshold.
        Then, we have to show the dendrogram plot. We have to capture the clickevent to set alternative treshold.
        We're doing this as long as we're not satisfied with the treshold. When the plot is closed, save the current
        treshold.
        """
        names = set(d.name for d in dists)
        classes = set(type(d) for d in dists)
        if len(names) != 1 or len(classes) != 1:
            raise TableException('Cannot group different distributions')
        name = names.pop()
        if name != trans.name():
            raise TableException('Transformations not matching "{}" and "{}"'
                                 .format(name, trans.name()))
        cls = classes.pop()
        if cls == features.MultiFeatureDistribution:
            sub_dists = zip(*[d.counts for d in dists])
            sub_trans = trans.trans
            sub_groups = [Table.group(metas, sd, st, norm=norm, out_path=out_path)
                          for sd, st in zip(sub_dists, sub_trans)]
            return ['*'.join([str(t) for t in sg]) for sg in zip(*sub_groups)]
        else:
            combinations = list(itertools.combinations(range(len(dists)), 2))
            distances = [dists[a].distance(dists[b], norm)
                         for (a, b) in combinations]
            linkage = cluster.linkage(distances)
            threshold = 0.1  # TODO dynamic threshold
            if out_path:
                figure = plt.figure(figsize=(10, 20))
                labels = ['{} ({})'.format(' '.join(m.source()), m.files[0].records)
                          for m in metas]
                tp = ThresholdPicker(figure, linkage, labels)
                show_dendrogram(linkage, labels, threshold)
                plt.title(name)
                #plt.tight_layout()
                tp.connect()
                plt.show()
                tp.disconnect()
                plt.close(figure)
                threshold = tp.threshold

                # plt.show() destroys the picture, draw again for saving
                figure = plt.figure(figsize=(10, 20))
                show_dendrogram(linkage, labels, threshold)
                plt.title(name)
                #plt.tight_layout()
                # TODO too long
                file_path = os.path.join(out_path, 'dendogram_{}_{}.pdf'.format(name, norm))
                # file_path = os.path.join(out_path, 'dendogram_{}_{}.pdf'.format('private', norm))
                plt.savefig(file_path, dpi=3000)
                plt.close(figure)
            groups = cluster.fcluster(linkage, threshold, criterion='distance')
            return list(groups)

    @staticmethod
    def marginalize_partial(metas, dists, trans, subspaces, output_path):
        """
        Compute marginal distributions for specific subspaces
        """
        marginal_trans = []
        for dimensions in subspaces:
            marginal_trans.append(trans.subspace(dimensions))
        independent = features.Append(marginal_trans)
        for d, meta in zip(dists, metas):
            marginal_dists = []
            for dimensions in subspaces:
                marginal_dists.append(d.subspace(dimensions))
            m = features.MultiFeatureDistribution(independent, marginal_dists)
            w = dataset.Writer(output_path, meta, skip_writing=True)
            w.save_distributions(m)
            w.close()
        # TODO output independent transformation
        print(independent.name())

    @staticmethod
    def marginalize(metas, dists, trans, output_path):
        independent = features.Append(trans.trans)
        for d, meta in zip(dists, metas):
            if isinstance(d, features.MultiDimDistribution):
                m = features.MultiFeatureDistribution(independent, d.marginal)
                w = dataset.Writer(output_path, meta, skip_writing=True)
                w.save_distributions(m)
                w.close()


class ThresholdPicker:

    def __init__(self, figure, linkage, labels):
        self.figure = figure
        self.linkage = linkage
        self.cid_press = None
        self.cid_release = None
        self.labels = labels
        self.threshold = None

    def connect(self):
        self.cid_press = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cid_release = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)

    def on_press(self, event):
        # TODO do all in on_release, skip if mouse dragged
        threshold = event.xdata
        if threshold is None or threshold < 0:
            return
        self.threshold = threshold
        plt.cla()
        show_dendrogram(self.linkage, self.labels, threshold)

    def on_release(self, event):
        self.figure.canvas.draw()

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.cid_press)
        self.figure.canvas.mpl_disconnect(self.cid_release)
