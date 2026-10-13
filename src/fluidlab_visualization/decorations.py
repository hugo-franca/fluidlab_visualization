import matplotlib.pyplot as plt
import numpy as np

def create_grid_legend(ax, grid_handles, column_labels = None, row_labels = None):
    ncol = grid_handles.shape[1]
    grid_handles = grid_handles.T.flatten()
    empty_labels = np.full(grid_handles.shape, r'')
    columnspacing = 3
    labelspacing = 0
    handletextpad = 0
    legend = ax.legend(grid_handles, empty_labels, ncol=ncol, loc='upper right', bbox_to_anchor=(1.0, 0.95), columnspacing=columnspacing, handletextpad=handletextpad, labelspacing=labelspacing, frameon=False)

    # Get legend bounding box
    fig = ax.figure
    fig.canvas.draw()

    bbox = legend.get_window_extent(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.transFigure.inverted())

    if( column_labels is not None ):
        x0, x1 = bbox.x0, bbox.x1
        pad_legend_left = 0.0
        pad_negative_legend_right = 0.0
        pad_middle = 0.01
        xs = np.linspace(x0 + pad_legend_left, x1 - pad_negative_legend_right, len(column_labels) + 2)
        for i, label in enumerate(column_labels):
            x = xs[i+1]
            fig.text(x + i*pad_middle, bbox.y1 + 0.01, label, ha='center', va='bottom')

    if( row_labels is not None ):
        # Row centers
        y0, y1 = bbox.y0, bbox.y1
        pad_fix = 0.01
        ys = np.linspace(y1, y0, len(row_labels) + 2)

        for i, label in enumerate(row_labels):
            fig.text(bbox.x0 - 0.01, ys[i+1], label, ha='right', va='center')

    return

