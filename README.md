# A demo of HDBSCAN oversensitivity

Interactively plots HDBSCAN clustering solutions given a ground truth of two real clusters of n_points points each (just simple Gaussians) and a variable number of noise points. You can also tweak the HDBSCAN settings live! Notice how smaller values of `min_cluster_size` and `min_samples`, while useful to detect smaller clusters, also drive up the false positive rate.

Matplotlib stuff is based on [this tutorial](https://matplotlib.org/3.1.1/gallery/widgets/slider_demo.html).

Other things to note:
* The data is 5D, but only two dimensions are displayed (making it more similar to Gaia data)
* Unclustered noise points (i.e. not assigned to a cluster by HDBSCAN) are displayed as faint black points; clustered points are displayed with colours. There are only 10 colours, so it will start reusing them eventually!
* There are way fewer points involved (a few thousand by default) vs. what you see in Gaia. You can turn up n_points massively to e.g. 2000 to simulate this better and see more false positives, but... good luck to your computer =) this example is contrived to run efficiently so don't expect values of min_cluster_size and min_samples to behave the same way they do for Gaia works.

## Quick example

This is what the clustering solution _should_ look like...

![Two Gaussians detected properly with no noise points in sight](plots/no_noise_small.jpg?raw=true "No noise points")

... but when `n_noise_points` >> `n_points` in each cluster (250 in this case), all hell breaks loose!

![Way too many noise points for HDBSCAN! Postprocessing is required!](plots/some_noise_small.jpg?raw=true "7500 noise points")

## Requirements
* Python 3.6+
* Numpy
* Matplotlib
* HDBSCAN

## Running

Tweak the settings at the top of the script if you'd like, and run it with

```
python3 hdbscan_issue_demo.py 
```

If you can't get plotting to work (make sure matplotlib is using a backend like Tkinter, and displaying in its own window), you can also try running the simple demo of sliders with

```
python3 slider_example.py 
```