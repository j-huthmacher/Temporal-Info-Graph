"""
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from scipy.special import softmax

from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

import scipy.ndimage as ndimage

plt.style.use('seaborn')

def plot_emb(x, y, dim=2, title="", use_pca = True, ax = None):
    """
    """
    if use_pca:
        pca = PCA(n_components=dim, random_state=123)
        x = pca.fit_transform(x)
    
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(12, 5))

    print(ax)

    sns.scatterplot(x[:,0], x[:,1], hue=y.astype(str), ax=ax[0],  legend=False)
    sns.countplot(x=y, ax=ax[1])

    # if ax is None:
    #     fig.suptitle(title)
    #     return fig

def class_contour(x, y, clf, precision = 0.02, title="", ax = None):
    """
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    if len(x.shape) != 2:
        x = np.array(list(x))
    if len(y.shape) != 2:
        y = np.array(list(y))

    if x.shape[1] > 2:
        pca = PCA(n_components=2, random_state=123)
        x = pca.fit_transform(x) 
        raise ValueError("Too much dimensions! (Only 2D model is possible)")


    h = precision  # step size in the mesh
    x_min, x_max = x[:, 0].min() - max(h, 0.05), x[:, 0].max() + max(h, 0.05)
    y_min, y_max = x[:, 1].min() - max(h, 0.05), x[:, 1].max() + max(h, 0.05)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

        
    Z = clf(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).detach().cpu().numpy()
   
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))  

    Z = np.argmax(Z, axis=1)
    # Z1 = Z[:, 0] * -1 # contour only for class 0
    Z = Z.reshape(xx.shape)

    Z = ndimage.gaussian_filter(Z, sigma=1.0, order=0)

    # z = z[:-1, :-1]
    # levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=sns.color_palette("Spectral", as_cmap=True))

    # Z2 = Z[:, 1] * -1 # contour only for class 1
    # Z2 = Z2.reshape(xx.shape)
    # ax.contourf(xx, yy, Z2, alpha=0.8, cmap="viridis")


    ax.scatter(x[:, 0], x[:, 1], c=y.astype(int), cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

    ax.set_title(title)

    if fig is not None:
        return fig

def eval_plots(folder, title="", precision = 0.02):
    """
    """
    data = np.load(f"{folder}embeddings.npz", allow_pickle=True)
    x = data["x"]
    y = data["y"]

    mlp_train_loss = np.load(f"{folder}TIG_MLP.train_losses.npy")
    mlp_val_loss = np.load(f"{folder}TIG_MLP.val_losses.npy")

    mlp_train_metrics = np.load(f"{folder}TIG_MLP.train.metrics.npy")
    mlp_val_metrics = np.load(f"{folder}TIG_MLP.val.metrics.npy")

    clf = torch.load(f"{folder}TIG_MLP.pt")

    # fig, ax = plt.subplots(2,2, figsize=(12, 10))
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)

    fig.suptitle(folder, fontsize=12)

    #### Plot Embeddings ####
    ax = [fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[0:2, 1])]
    plot_emb(x, y, use_pca=(x.shape[1] > 2), ax=ax)
    ax[0].set_title(f"Embeddings in 2D space {'(PCA used)'if (x.shape[1] > 2) else ''}")
    ax[1].set_title("Class distribution")

    #### Plot Contour ####
    ax = fig.add_subplot(gs[2:, 0])
    class_contour(x, y, clf, precision, ax = ax)
    ax.set_title("MLP Decision Boundaries")

    #### Loss Curve ####
    ax = fig.add_subplot(gs[2, 1])
    ax.axhline(min(mlp_train_loss.min(), mlp_val_loss.min()), 0,
                    mlp_train_loss.shape[0], lw=1, ls=":", c="grey")
    ax.plot(mlp_train_loss, label="Trainings Loss")
    ax.plot(mlp_val_loss, label="Validation Loss")    
    
    ax.set_yticks(list(ax.get_yticks()) + [min(mlp_train_loss.min(), mlp_val_loss.min())])

    ax.set_title("MLP Loss")
    ax.legend()

    #### Accuracy Curve ####
    ax = fig.add_subplot(gs[3, 1])

    ax.axhline(max(mlp_train_metrics[:, 0].max(), mlp_val_metrics[:, 0].max()), 0,
                   mlp_train_metrics[:, 0].shape[0], lw=1, ls=":", c="grey")
    ax.axhline(max(mlp_val_metrics[:, 0].max(), mlp_val_metrics[:, 0].max()), 0,
                   mlp_val_metrics[:, 0].shape[0], lw=1, ls=":", c="grey")

    ax.plot(mlp_train_metrics[:, 0], label="Top-1 (Train)")
    ax.plot(mlp_val_metrics[:, 0], label="Top-1 (Validation)")    
    
    ax.set_yticks(list(ax.get_yticks()) + [max(mlp_train_metrics[:, 0].max(), mlp_val_metrics[:, 0].max())])
    ax.set_yticks(list(ax.get_yticks()) + [max(mlp_val_metrics[:, 0].max(), mlp_val_metrics[:, 0].max())])

    ax.set_title("MLP Accuracy")
    ax.legend()

def plot_desc_loss_acc(x, y, clf, loss, metric, prec = 0.02, title="", n_epochs=None):
    """
    """
    fig = plt.figure(figsize=(9, 4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    fig.suptitle(title, fontsize=12)

    #### Plot Contour ####
    ax = fig.add_subplot(gs[0:, 0])
    class_contour(x, y, clf, prec, ax = ax)
    ax.set_title("MLP Decision Boundaries")


    #### Loss Curve ####
    if isinstance(loss, list):
        loss = np.array(loss)

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(loss.min(), 0,
               loss.shape[0], lw=1, ls=":", c="grey")

    ax.plot(loss)
    
    if n_epochs is not None:
        ax.set_xlim(0, n_epochs)

    ax.text(x=ax.get_xlim()[1] + 0.5, y=loss.min(), s='%.3f' % loss.min(),  va="center")

    # ax.set_yticks(list(ax.get_yticks()) + [loss.min()])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title("MLP Loss")
    # ax.legend()

    #### Accuracy Curve ####
    if isinstance(metric, list):
        metric = np.array(metric)

    ax = fig.add_subplot(gs[1, 1])

    ax.axhline(metric.max(), 0,
               metric.shape[0], lw=1, ls=":", c="grey")

    
    ax.plot(metric)
    
    if n_epochs is not None:
        ax.set_xlim(0, n_epochs)
    
    ax.text(x=ax.get_xlim()[1] + 0.5, y=metric.max(), s='%.3f' % metric.max(), va="center")

    # ax.set_yticks(list(ax.get_yticks()) + [metric.max()])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title("MLP Metric (Accuracy)")
    # ax.legend()

    return fig