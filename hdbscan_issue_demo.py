import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------------------------------
# SETTINGS
# ---------------------------------------
# Controls number of points in cluster. Set lower if it's running too slow (m_clSize will need to be lower too)
# If you set it much higher, you may also want to reduce the alpha of noise points in noise_kwargs
n_points = 250
max_n_points = 500
max_noise_points = max_n_points * 20
n_noise_points = 0
n_dimensions = 5  # 5 is most analogous to Gaia data
gaussian_kwargs = dict(scale=0.02)  # Size of the Gaussian used for the real clusters

# HDBSCAN settings
min_cluster_size = 20
min_samples = 10
min_cluster_size_range = (10, 50)
min_samples_range = (10, 50)
hdbscan_kwargs = dict(core_dist_n_jobs=1, cluster_selection_method='leaf', allow_single_cluster=False)

# Stuff for the figure (tweak if your screen is very different)
figsize = (8, 6)
dpi = 100

line_kwargs = dict(ms=4, alpha=1.0)
noise_kwargs = dict(ms=2, alpha=0.2, color='k')

# ---------------------------------------
# DATA
# ---------------------------------------
# Change this if you want to play around with a different model
random = np.random.default_rng(seed=42)
data_noise = random.uniform(size=(max_noise_points, 5))
data_cluster = random.normal(size=(max_n_points, 5), loc=0.3, **gaussian_kwargs)
data_cluster_2 = random.normal(size=(max_n_points, 5), loc=0.7, **gaussian_kwargs)

# Combine them in an alternating way
data_both = np.ravel([data_cluster, data_cluster_2], order="F").reshape(n_dimensions, max_n_points*2).T

# ---------------------------------------
# MATPLOTLIB SETUP
# ---------------------------------------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
plt.subplots_adjust(bottom=0.26, top=0.95)
ax.set(xlabel='dimension 0', ylabel='dimension 1', xlim=(0, 1), ylim=(0, 1))

# Add sliders
axcolor = 'lightgoldenrodyellow'
ax_points = plt.axes([0.25, 0.14, 0.55, 0.03], facecolor=axcolor)
ax_noise_points = plt.axes([0.25, 0.10, 0.55, 0.03], facecolor=axcolor)
ax_min_cluster_size = plt.axes([0.25, 0.06, 0.55, 0.03], facecolor=axcolor)
ax_min_samples = plt.axes([0.25, 0.02, 0.55, 0.03], facecolor=axcolor)

slider_points = Slider(ax_points, 'n_points',
                       25, max_n_points, valinit=n_points, valstep=25)
slider_noise_points = Slider(ax_noise_points, 'n_noise_points',
                             0, max_noise_points, valinit=n_noise_points, valstep=25)
slider_min_cluster_size = Slider(ax_min_cluster_size, 'min_cluster_size',
                                 *min_cluster_size_range, valinit=min_cluster_size, valstep=5)
slider_min_samples = Slider(ax_min_samples, 'min_samples', *min_samples_range, valinit=min_samples, valstep=5)

# Add a reset button
ax_reset = plt.axes([0.85, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', color=axcolor, hovercolor='0.975')

# Add empty lines to update later
lines = [ax.plot([], [], 'o', zorder=100, **line_kwargs)[0] for x in range(10)]
line_noise = ax.plot([], [], 'o', zorder=-100, **noise_kwargs)[0]


# ---------------------------------------
# MATPLOTLIB UPDATE FUNCTIONS
# ---------------------------------------
def update(val):
    n_point = int(slider_points.val)
    n_noise = int(slider_noise_points.val)
    min_cluster = int(slider_min_cluster_size.val)
    min_samp = int(slider_min_samples.val)

    print(f"\rPlotting two {n_point} size clusters, {n_noise} noise points with "
          f"min_cluster_size={min_cluster}, min_samples={min_samp}!", end="")

    # Firstly, run HDBSCAN
    data = np.vstack((data_both[:n_point * 2], data_noise[:n_noise]))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samp, **hdbscan_kwargs)
    labels = clusterer.fit_predict(data)
    unique_labels = np.unique(labels)
    good_labels = unique_labels[unique_labels != -1]

    # Next, update all the lines
    # The noise line is easy
    label_noise = labels == -1
    line_noise.set_data(data[label_noise, 0], data[label_noise, 1])

    # Coloured points are much harder, as we may need to plot multiple clusters as the same colour if there are > 10
    # So firstly, decide which of the ten lines every point should belong to...
    matches_this_label = np.zeros((len(lines), data.shape[0]), dtype=bool)
    for a_label in good_labels:
        line_to_edit = a_label % 10
        matches_this_label[line_to_edit] = np.logical_or(labels == a_label, matches_this_label[line_to_edit])

    # And secondly, update them
    for i_line, a_line in enumerate(lines):
        a_line.set_data(data[matches_this_label[i_line], 0], data[matches_this_label[i_line], 1])

    # Redraw canvas!
    ax.set_title(f"{len(good_labels)} clusters reported by HDBSCAN (correct value: 2)")

    fig.canvas.draw_idle()
    print("\rDone", end="")


# Setup calling of update()
slider_points.on_changed(update)
slider_noise_points.on_changed(update)
slider_min_cluster_size.on_changed(update)
slider_min_samples.on_changed(update)


# ---------------------------------------
# RESET BUTTON
# ---------------------------------------
def reset(event):
    slider_points.reset()
    slider_noise_points.reset()
    slider_min_cluster_size.reset()
    slider_min_samples.reset()


button.on_clicked(reset)

plt.show()
update(None)
