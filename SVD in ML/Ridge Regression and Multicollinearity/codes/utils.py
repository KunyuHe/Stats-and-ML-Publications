import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import Ridge

title_font = FontProperties(family="Arial", size=14, weight="semibold")
axis_font = FontProperties(family="Arial", size=12)
ticks_font = FontProperties(family="Arial", size=10)


def corr_plot(ax, corr, labels, plot_title="", cmap="coolwarm",
              triangle=False, upper=True, annot_size=20, decimals=2,
              partial=False, xlabs=None, ylabs=None):
    """
    Plot a correlation matrix.
    Inputs:
        - ax: (Axes) instance to draw the plot on
        - corr: (2D-array) the correlation matrix
        - labels: ([str]) names of the variables for plotting
        - sub: (Bool) whether the plot is a subplot, control for the title font
        - plot_title: (string) title of the plot
        - cmap: (string) color map for the heat map
        - triangle: (Bool) whether to plot the upper or lower part of the matrix
            as it's symmetric
        - upper: (Bool) if only plot half of the plot, whether it's upper
        - font_size: (int, int, int) font size for title, axis, and tick
        - annot_size: (int) font size for the correlation number annotations
        - decimals: (int) round the correlation coefficients to
    """
    default = {'ax': ax,
               'cmap': cmap, 'cbar_kws': {"shrink": .5},
               'vmax': 1, 'vmin': -1, 'center': 0,
               'square': True, 'linewidths': 0.5}
    if annot_size != 0:
        default['annot'] = True
        default['annot_kws'] = {"size": annot_size}
    corr = corr.round(decimals)

    if triangle:
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[[np.triu_indices_from, np.tril_indices_from][int(upper)](
            mask)] = True
        sns.heatmap(corr, mask=mask, **default)
    else:
        sns.heatmap(corr, **default)

    # Make tick labels to
    if not partial:
        xlabs, ylabs = labels, labels
    ax.set_title(plot_title, fontproperties=title_font)
    ax.set_xticklabels(xlabs, fontproperties=axis_font, rotation='vertical')
    ax.set_yticklabels(ylabs, fontproperties=axis_font, rotation='horizontal')
    ax.tick_params(axis='both', length=0.0)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(cbar.ax.get_yticks(), fontproperties=ticks_font)
    cbar.ax.set_title('Correlation', fontproperties=ticks_font)

    if partial:
        return None

    if upper:
        x_hide, y_hide = 0, -1
        ax.xaxis.tick_top()
    else:
        x_hide, y_hide = -1, 0
    plt.setp(ax.get_xticklabels()[x_hide], visible=False)
    plt.setp(ax.get_yticklabels()[y_hide], visible=False)


def plot_ridge_coefs(ax, X_train, y_train, labels,
                     alpha_lst, s_lst, c_lst,
                     format_y="{:.2f}"):
    """
    Plot the coefficients of a set of ridge regression models.
    - ax: (Axes) instance to draw the plot on
    - X_train: (2D-array) the training feature matrix
    - y_train: (2D-array) the training target vector
    - labels: ([string]) list of feature names
    - alpha_lst: ([float]) list of penalty strengths (lambda)
    - s_lst: ([string]) list of matplotlib marker shapes
    - c_lst: ([string]) list of marker hex colors
    - format_y: (str) formatting string for the y-axis
    """
    for alpha, s, c in zip(alpha_lst, s_lst, c_lst):
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        ax.scatter(range(len(labels)), ridge.coef_, marker=s, s=40, color=c,
                   label="Ridge, $\lambda = %s$" % alpha, edgecolor='black')

    ax.set_title("Ridge Regression Coefficients", fontproperties=title_font)
    ax.set_ylabel("Coefficients", fontproperties=axis_font)
    ax.set_yticklabels([format_y.format(val) for val in ax.get_yticks()],
                       fontproperties=ticks_font)
    ax.set_xlabel("Features", fontproperties=axis_font)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontproperties=ticks_font, rotation='vertical')
    ax.set_xlim(-0.1, len(labels)-0.9)
    ax.legend(ncol=2, loc=(0, 1.02))
