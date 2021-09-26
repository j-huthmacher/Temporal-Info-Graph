""" Script to create plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import scipy.ndimage as ndimage

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.style.use('seaborn')

#### Single Plots ####


def plot_performance_matrix(df: pd.DataFrame, y: [str], x: [str], value: str,
                            ax: matplotlib.axes.Axes = None, ytable: bool = True,
                            xtable: bool = True, xrowlabel: bool = True,
                            with_divider: bool = True):
    """ Function to plot performance matrix.

        Args:
            df: pd.DataFrame
                Pandas Dataframe with the values for the matrix.
            y: [str]
                List of column names that are used to group the data along
                the y axis (those column names appear on the y axis as labels).
            x: [str]
                List of columns names that are used to group the data along the
                x axis (those column names appear on the x axis as a labels).
            value: str
                Column name of the column that contains the value that is shown
                in each matrix cell.
            ax: matplotlib.axes.Axes (optional)
                Matplotlib axes at which the plot is created. If it is None
                a new axes is created.
            xrowlabel: bool
                Flag to determine if the table at the x axis contains
                row labels.
            with_divider: bool
                Determines if the matrix contains visual dividers.

        Return:
            matplotlib.image.AxesImage, pd.DataFrame
    """
    df = df.pivot_table(index=y, columns=x, values=value, dropna=False)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    max_val = max(np.nanmax(df.values), np.abs(np.nanmin(df.values)))
    im = ax.imshow(df, cmap="RdYlGn", vmin=-max_val, vmax=max_val)

    if with_divider:
        # The dividers have to be adapted if the groups that are used for
        # the matrix change.

        # ax.axhline(2.5, 0, lw=5, ls="-", c="white")
        ax.axhline(.5, 0, lw=5, ls="-", c="white")

        ax.axvline(2.5, 0, lw=5, ls="-", c="white")
        ax.axvline(5.5, 0, lw=5, ls="-", c="white")

        ax.axvline(2.5, 0, lw=1, ls="--", c="grey", clip_on=False)
        ax.axvline(5.5, 0, lw=1, ls="--", c="grey", clip_on=False)

        # ax.axhline(2.5, 0, lw=1, ls="--", c="grey", clip_on=False)
        ax.axhline(.5, 0, lw=1, ls="--", c="grey", clip_on=False)

    if xtable:
        # As x label a table is used, since the values in a cell corresponds to a group
        cellText = np.array(list(df.columns.values), dtype=object).T
        table = ax.table(cellText=cellText if len(cellText.shape) > 1 else [cellText],
                         rowLabels=x if xrowlabel else [""] * len(x),
                         cellLoc="center",
                         loc='bottom',
                         edges='open')

        table.scale(1, 5.5)
        table.set_fontsize("large")

        for cell in table._cells:
            if cell[1] >= 0:
                table._cells[cell].get_text().set_rotation(90)

        table.auto_set_font_size(False)
        if with_divider:
            ax.axvline(2.5, -1., 0, lw=1, ls="--", c="grey", clip_on=False)
            ax.axvline(5.5, -1., 0, lw=1, ls="--", c="grey", clip_on=False)

    if ytable:
        # As x label a table is used, since the values in a cell corresponds to a group
        cellText = np.array(list(df.index.values))
        table = ax.table(cellText=cellText if len(cellText.shape) > 1 else np.array([cellText]).T,
                         colLabels=y,
                         cellLoc='right',
                         edges='open',
                         bbox=[-0.6, 0.01, 0.6, 1.1])
        table.set_fontsize("large")
        table.auto_set_font_size(False)
        table.auto_set_column_width(col=list(range(len(y))))

        if with_divider:
            ax.axhline(2.5, -0.6, 0, lw=1, ls="--", c="grey", clip_on=False)
            ax.axhline(.5, -0.2, 0, lw=1, ls="--", c="grey", clip_on=False)

    # Remove default labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.grid(False)

    for x_val in ax.get_xticks():
        for y_val in ax.get_yticks():
            text = (""
                    if np.isnan(df.iloc[y_val, x_val])
                    else "%.2f" % df.iloc[y_val, x_val])
            ax.text(x_val, y_val, text, va='center', ha='center')

    if fig is not None:
        fig.colorbar(im)
        fig.tight_layout()

    return im, df


def plot_metric(metrics: np.ndarray, path: str, num_lines: int = 3, orientation: str = "h",
                ax: matplotlib.axes.Axes = None):
    """ Function to plot the metric values as line plot over some period of time, e.g. epochs.

        Args:
            metrics: np.ndarray
                1D numpy array, where each entry has the form
                (EXPERIMENT_NAME, [AUC_MEAN], [AUC_STD], [PREC_MEAN], [PREC_AUC]) with the mean
                and standard deviation ove the repetitions of an experiment.
            path: str
                Location where the plot is stored.
            num_lines: int
                Number of different lines within a plot, i.e. how many different experiments
                are visualized.
            orientation: str
                Orientation of the plot, i.e. if the metric (e.g. AUC) is located at the
                y axis or the y axis. Default "h" ("v" is the other option), which means the
                metric is placed the y axis.
            ax: matplotlib.axes.Axes (optional)
                Matplotlib axes at which the plot is created. If it is None
                a new axes is created.
    """
    plt.style.use("seaborn-talk")
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    if ax is None:
        if orientation == "v":
            fig, ax = plt.subplots(
                len(metrics) // num_lines, 1, figsize=(4, len(metrics) * 6))
        else:
            fig, ax = plt.subplots(1, len(metrics) // num_lines, figsize=(len(metrics) * 6, 4),
                                   sharey=True)
    else:
        fig = ax[0].figure
    ax = np.array(ax).flatten()

    for i, (name, val, val_std) in enumerate(metrics):
        i_mod = i % num_lines
        i = i // num_lines

        p = ax[i].plot(np.arange(val.shape[0]), val, label=name)
        cc = p[-1].get_color()
        ax[i].axhline(np.mean(val[-20:]), 0, val.shape[0], lw=2, ls=":", c=cc,
                      label='Mean*: %.3f, Std.*: %.3f' % (np.mean(val[-20:]),
                                                          np.mean(val_std[:-20])))

        # Display variation as filled area. Use only the last 20 values.
        ax[i].fill_between(np.arange(val.shape[0]), val - val_std, val + val_std,
                           color=cc, alpha=0.5)
        # Display variation of the mean as filled area. Use only the last 20 values.
        ax[i].fill_between([0, val.shape[0] - 1],
                           np.mean(val[-20:]) - np.mean(val_std[-20:]),
                           np.mean(val[-20:]) + np.mean(val_std[-20:]),
                           color=cc, alpha=0.25)

        # ax[0].set_ylabel("AUC")
        # ax[i].set_xlabel("Epochs")

        # ax[i].legend(loc='center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=2)
        # ax[i].margins(x=.05)

    # ax[-1].set_xlabel("Epochs")

    # for a in ax:
        # h, l = a.get_legend_handles_labels()
        # a.legend([*h[0::2], *h[1::2]],
        #          [*l[0::2], *l[1::2]],
        #          loc='center', bbox_to_anchor=(0.5, -0.3),
        #          fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    return fig


def plot_emb(x: np.array, y: np.array, title: str = "", sup_title: str = "",
             ax: matplotlib.axes.Axes = None, count: bool = False, mode: str = "PCA",
             figsize=(7, 7), legend: bool = False):
    """ Plot embeddings.

        Args:
            x: np.array (could also be torch.Tensor)
                Features of the embeddings (i.e. coordinates).
            y: np.array (could also be torch.Tensor)
                Labels corresponding to the features.
            title: str
                Title of the plot.
            ax: matplotlib.axes.Axes
                Axes for the plot. If None a new axis is created.
            count: bool
                Flag to decide if additionally the counts are plotted beside the embedding plot.
            mode: str
                Approach that is used to plot high dimensional data.
                Options: ["Plain", "PCA", "TSNE"]
            figsize: tuple
                Size of the matplotlib figure.
            legend: bool
                Determines if the legend of the plot is activated.
        Return:
            matplotlib.figure.Figure: Figure of the plot IF the axis paramter is None.
            Otherwise nothing is returned.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    mode_str = ""

    if mode == "Plain":
        # For the case that the emebeddings only have two dimensions.
        x = x[:x.shape[1]]
        x_args = {"xticks": [],
                  "yticks": [],
                  "xlabel": "Emb. Features",
                  "ylabel": "Samples",
                  "title": f"Embeding Values (first {x.shape[1]} samples)"
                  }

        _ = plot_heatmap(x, im_args=x_args, figsize=figsize)
        return

    if x.shape[1] > 2:
        # For the case we have high dimensional embeddings.
        mode_str = f"({mode})"
        if mode == "PCA":
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(x)
        elif mode == "TSNE":
            pca = TSNE(n_components=2, random_state=123)
            x = pca.fit_transform(x)

    fig = None
    if ax is None:
        subplot_args = {"nrows": 1,
                        "ncols": 2 if count else 1,
                        "figsize": (12, 6) if count else figsize
                        }
        fig, ax = plt.subplots(**subplot_args)

    # Method to easily iterate over axes of subplots.
    ax = np.squeeze([ax]) if np.squeeze([ax]).shape else [ax]

    # Coloring based on the labels to see if emebddings from the same class form clusters.
    cmap = matplotlib.cm.get_cmap('Spectral')
    norm = matplotlib.colors.Normalize(vmin=np.min(y), vmax=np.max(y))
    y = y.astype(int)

    for label in np.unique(y):
        ax[0].scatter(x[y == label][:, 0], x[y == label][:, 1],
                      color=cmap(norm(label)), label=label, edgecolors='w')

    if legend:
        ax[0].legend()

    ax[0].set_title(f"{title} {mode_str}")

    ax[0].set_xlim(-5, 5)
    ax[0].set_ylim(-5, 5)

    if count:
        # Adds count plot of the labels beside the embeddings.
        sns.countplot(x=y, ax=ax[1])

    if fig is not None:
        fig.suptitle(sup_title)
        return fig


def plot_emb_pred(x: np.ndarray, y: np.ndarray, clf: torch.nn.Module, precision: float = 0.02,
                  title: str = "", ax: matplotlib.axes.Axes = None):
    """ Plots embedding plus the prediction by the given classifier. If the embedding dimension is
        2 the decision boundaries are plotted as well. The prediction is visually shown by the
        border of the dots.

        Important: For the decision boundaries we do the inference for each point in the grid, which
        can be computationally expensive depending on the grid size (precision).

        Args:
            x: np.array (could also be torch.Tensor)
                Features of the embeddings (i.e. coordinates).
            y: np.array (could also be torch.Tensor)
                Labels corresponding to the features.
            clf: torch.nn.Module
                PyTorch classifier that is used to predict the class labels for the given features.
            precision: float
                Precision of the grid that is created to visualize the class boundaries.
            title: str
                Title of the plot
            ax: matplotlib.axes.Axes
                Axes for the plot. If None a new axis is created.
        Return:
            matplotlib.figure.Figure: Figure of the plot IF the axis paramter is None.
            Otherwise nothing is returned.
    """
    #### Some Preparations ####
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if len(x.shape) != 2:
        x = np.array(list(x))
    if len(y.shape) != 2:
        y = np.array(list(y))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    cmap = sns.color_palette("Spectral", as_cmap=True)

    if x.shape[1] <= 2:
        #### Create a grid for the decision boundaries ####
        h = precision  # Grid precision / stepsize
        x_min, x_max = x[:, 0].min() - max(h, 0.05), x[:, 0].max() + max(h, 0.05)
        y_min, y_max = x[:, 1].min() - max(h, 0.05), x[:, 1].max() + max(h, 0.05)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        device = clf.device
        clf = clf.cpu()

        # Prediction for the grid fields
        Z = clf(torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                dtype=torch.float32)).detach().cpu().numpy()
        clf = clf.to(device)

        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        # Make boundaries a bit smoother
        Z = ndimage.gaussian_filter(Z, sigma=1.0, order=0)

        #### Plot Decision Boundaries ####
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    else:
        #### The Embeddings are high dimensions (i.e. > 2) ####
        yhat = clf(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
        pca = PCA(n_components=2, random_state=123)
        x = pca.fit_transform(x)

        #### Plot Predicted Class ####
        ax.scatter(x[:, 0], x[:, 1], c=np.argmax(yhat, axis=1).astype(int),
                   cmap=cmap,
                   facecolors='none', s=80, linewidth=2)

    ax.scatter(x[:, 0], x[:, 1], c=y.astype(int), cmap=cmap, edgecolors='w')

    ax.set_title(title)

    if fig is not None:
        return fig


def plot_curve(data: dict, ax: matplotlib.axes.Axes = None, n_epochs: int = None, title: str = "",
               model_name: str = "TIG", line_mode: callable = np.min, n_batches: int = None,
               l_pos: str = "right"):
    """ Plot curve, e.g. loss or metric.

        Args:
            data: dict
                Dictionary containing the data that should be plotted.
                Key is used as label (multiple keys are possible) and the value contains
                the numerical values for the plot.
            ax: matplotlib.axes.Axes
                Axes for the plot.
            n_epochs: int
                Number of epochs. If provided it is used to determine the x-axis ticks.
            title: str
                Title of the plot.
            model_name: str
                Name of the model that corresponds to the data.
            line_mode: callable
                Function that is used to generate a constant horizontal line.
                Possibilities: [np.min, np.max]
            n_batches: int
                Number of batches. If provided it is used to determine the x-axis ticks.
            l_pos: str
                String to position the legen. Options ["below"]
        Return:
            matplotlib.figure.Figure: Figure of the plot.
    """
    legend_pos = {
        "loc": 'upper left',
        "bbox_to_anchor": (1, 1)
        }

    if l_pos == "below":
        legend_pos = {
            "loc": 'upper left',
            "bbox_to_anchor": (0, -0.2)
            }
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
        legend_pos = {
            "loc": 'upper center',
            "bbox_to_anchor": (0.5, -0.2)
        }

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for _, name in enumerate(data):
        line_data = data[name]
        if isinstance(line_data, list):
            line_data = np.array(line_data)

        p = ax.plot(line_data, label='%.3f (%s)' % (line_data[-1], name))
        ax.axhline(line_mode(line_data), 0, line_data.shape[0], lw=1, ls=":", c=p[-1].get_color(),
                   label='%.3f (%s)' % (line_mode(line_data), name))

        if n_epochs is not None:
            # Set the ticks based on the epochs.
            ax.set_xlim(0, n_epochs - 1 if n_batches is None else (n_batches * n_epochs) - 1)
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title(f"{model_name} {title}")
    ax.figure.tight_layout()

    _, labels = ax.get_legend_handles_labels()
    # rows = 4 if len(labels) > 6 else 2
    ax.legend(**legend_pos, fancybox=True, shadow=True, ncol=2)

    return ax.figure


def plot_heatmap(matrix: np.ndarray, xlabel: str = "", ylabel: str = "", num_ticks: tuple = None,
                 cbar_title: str = "", im_args: dict = {}, ax: matplotlib.axes.Axes = None,
                 figsize: tuple = (5, 5)):
    """ Creates a single heatmap for a given matrix.

        Paramter:
            matrix: np.ndarray
                Numpy array representing the matrix that is shown as heatmap
            xlabel: str
                Label for the x axis.
            ylabel: str
                Label for the y axis.
            num_ticks: tuple
                Tuple of int containing at the first position the number of ticks for the x axis and
                at the second position the number of ticks for the y axis.
            cbar_title: str
                Title for the colorbar.
            img_args: dict
                Dictionary with arguments for the axis. This arguments are forwarded
                to ax.set(**im_args).
            ax: matplotlib.Axis
                Axis object if the heatmap should be plotted at a already existing axis.
        Return:
            matplotlib.Figure: Figure containing grid of heatmaps.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    hm = ax.imshow(matrix, cmap="YlGn")

    cbar = ax.figure.colorbar(hm, ax=ax, orientation="horizontal")
    cbar.ax.set_title(cbar_title)

    if num_ticks is None:
        xticks = matrix.shape[1]
        yticks = matrix.shape[0]
    else:
        xticks, yticks = num_ticks

    ax.set_xticks(np.arange(xticks))
    ax.set_yticks(np.arange(yticks))

    ax.set_xticklabels(np.arange(xticks))
    ax.set_yticklabels(np.arange(yticks))

    ax.set(**im_args)

    ax.grid(False)

    return ax.figure


def plot_skeleton(x: np.ndarray, y: np.ndarray, A: np.ndarray, ax: matplotlib.axes.Axes = None):
    """ Function to plot a skeleton based on the coordinates and an adjacency matrix.

        Args:
            x: np.ndarray
                X coordinates of the skeleton.
            y: np.ndarray
                Y coordinates of the skeleton.
            A: np.ndarray
                Adjacency matrix of the skeleton.
            ax: matplotlib.Axis
                Axis object if the heatmap should be plotted at a already existing axis.
        Return:
            matplotlib.Figure, matplotlib.axes.Axes: Figure and axes of the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Plot edges
    lines = []
    for i in range(0, len(x)):  # pylint: disable=consider-using-enumerate
        for e_x, e_y in zip(x[np.where(A[i] == 1)], y[np.where(A[i] == 1)]):
            lines.append(ax.plot([x[i], e_x], [
                         y[i], e_y], linestyle='-', linewidth=0.5, color='#c4c4c4', markersize=0))

    # Plot nodes
    ax.scatter(x, y, marker='o', linewidth=0, zorder=3)

    return ax.figure, ax
