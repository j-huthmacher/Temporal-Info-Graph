""" Some animation helper function to visualize the dynamic graph.

    Developed to be executed from a jupyter notebook.

    @author: jhuthmacher
"""
from typing import Any

import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation
import numpy as np
import pandas as pd

from PIL import Image

from copy import copy, deepcopy
from data import KINECT_ADJACENCY

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 100  
plt.style.use('seaborn')

images = []

def create_gif(fig, path, fill=True):
    """
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    im = Image.open(buf)

    images.append(im)

    if not fill:
        im.save(fp=path, format='GIF', append_images=images,
                save_all=True, duration=200, loop=0)
        fig.savefig(path.replace(".gif", ".final.png"), dpi=150)                
        images.clear()

    

    # # filepaths
    # fp_in = "/path/to/image_*.png"
    # fp_out = "/path/to/image.gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # img.save(fp=fp_out, format='GIF', append_images=imgs,
    #          save_all=True, duration=200, loop=0)

def animate_skeleton(data: pd.DataFrame, annot: bool = False, lim_frames: Any = None):
    """ Function to animate a skeleton.

        Important: The sekeleton is created based on a predefined adjacency matrix (data.KINECT_ADJACENCY)

        Parameters:
            data: pd.DataFrame
                Data frame contains the data after the preprocessing step.
            annot: bool
                Determines if the joints/nodes should be annotated with numbers.
            lim_frames: None or int
                Determines if the animated frames should be limited. For instance, if 
                lim_frames = 10 only the first 10 frames are animated.
        Return:
            matplotlib.animation            
    """
    A = KINECT_ADJACENCY

    nodes = 18
    fig, ax = plt.subplots(figsize=(8,4))   

    x = []
    y = []
    annotations = []
    try:
        for i, (_, val) in enumerate(data["frames"][0]["skeletons"][0].items()):
            x.append(val["x"])
            y.append(val["y"]) 
            annotations.append(ax.annotate(i, (val["x"], val["y"])))
    except:
        # Ignore empty skeletons
        pass
    
    x = np.asarray(x)
    y = np.asarray(y)

    # Edges
    lines = []
    for i in range(0, len(x)):
        for e_x, e_y in zip(x[np.where(A[i]==1)], y[np.where(A[i]==1)]):
            lines.append(ax.plot([x[i],e_x], [y[i], e_y], linestyle='-', linewidth=0.5, color='#c4c4c4', markersize=0))

    # Nodes
    nodes = ax.scatter(x, y, marker='o', linewidth=0, zorder=3)

    ax.set_title(f'{data["label"]} Video: https://www.youtube.com/watch?v={data["id"].replace(".json", "")}')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,0)

    ##################################################
    # Animation function that is called sequentially #
    ##################################################
    def animate(f):
        if f == 0 or len(data["frames"][f]["skeletons"]) == 0:
            return nodes, 

        annotations_empty = len(annotations) == 0
        lines_empty = len(lines) == 0

        x = []
        y = []
        try:
            for i, (_, val) in enumerate(data["frames"][f]["skeletons"][0].items()):
                x.append(val["x"])
                y.append(val["y"])
                if annotations_empty:
                    annotations.append(ax.annotate(i, (val["x"], val["y"])))
                else:
                    annotations[i].set_text(i)
                    annotations[i].set_position((val["x"], val["y"]))    

            nodes.set_offsets(np.c_[x, y])

            x = np.asarray(x)
            y = np.asarray(y)            
            
            if lines_empty:
                for i in range(0, len(x)):
                    for e_x, e_y in zip(x[np.where(A[i]==1)], y[np.where(A[i]==1)]):
                        lines.append(ax.plot([x[i],e_x], [y[i], e_y], linestyle='-', linewidth=0.5, color='#c4c4c4', markersize=0))
            else:
                j = 0
                for i in range(0, len(x)):
                    for e_x, e_y in zip(x[np.where(A[i]==1)], y[np.where(A[i]==1)]):
                        lines[j][0].set_ydata([y[i], e_y])
                        lines[j][0].set_xdata([x[i],e_x])                    
                        j += 1
            
            if len(lines) == 0 and len(x) == 0:
                for i in range(0, len(x)):
                    for e_x, e_y in zip(x[np.where(A[i]==1)], y[np.where(A[i]==1)]):
                        lines.append(ax.plot([x[i],e_x], [y[i], e_y], linestyle='-', linewidth=0.5, color='#c4c4c4', markersize=0))

        except Exception as e:#
            # print(e)
            # Not skeleton available! Do nothing.
            return nodes, 

        annotations_empty = len(annotations) == 0
        lines_empty = len(lines) == 0

        return nodes,
    
    ################
    # Animate plot #
    ################
    plt.close(fig)
    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=200, blit=True, repeat=False,
                                             frames= len(data["frames"]) if lim_frames is None else lim_frames)

    # ani.save('dissemination_example.gif', writer='imagemagick', fps=1)

    return ani