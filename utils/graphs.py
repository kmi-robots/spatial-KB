import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G,pos,connectionstyle='arc3, rad = 0.1',with_labels=True)
    #nx.draw_networkx_edges(G,pos,connectionstyle='arc3, rad = 0.1')
    #node_labels = nx.get_node_attributes(G, 'name')
    #nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'QSR')
    #modified built-in method in nx because it needs unique keys, i.e., fails for multi-graph
    draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #edge_labels = nx.get_edge_attributes(G, 'ext')
    #draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.draw()
    plt.show()


def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
):
    """Draw edge labels.
    Extended from original Networkx method to use with MultiDGraph,
    also refer to the official examples and docs at
    https://networkx.github.io/documentation/latest/auto_examples/index.html
    """

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2,_), label in labels.items(): #only modification
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=True,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items