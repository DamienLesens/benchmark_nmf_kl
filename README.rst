
Benchmark Repository for Nonnegative Matrix Factorization with KL loss
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of Nonnegative Matrix Factorization with KL loss:


$$\\min_{W\\in \\mathbb{R}^{m\\times r}_+, H\\in \\mathbb{R}^{r\\times n}_+} KL(X, WH)$$


where $m, n$ stand for respectively for the number of rows and columns of the data matrix $X$ which may have negative entries, 

$$X \\in \\mathbb{R}^{m \\times n}$$

In short, matrix $X$ is approximated by a low rank matrix $WH$ where each low-rank factor $W$ and $H$ have nonnegative entries, which makes NMF a part-based decomposition.

The rank for the NMF must be provided in the dataset. Several values may be specified, but the responsability of chosing candidate rank values by default does not fall on the solvers, nor the objective.

Install
--------

Create a conda environment from the ``environment.yml`` file.

Then specify in a config file which algorithms an datasets you want to use. An example is provided in ``example_config.yml``.


You can then run the benchmark the following commands:

.. code-block::

   benchopt run --config example_config.yml --output output_file


Use ``benchopt run -h`` for more details about options, or visit https://benchopt.github.io/.

You can plot specific curves using ``benchopt plot``.

Datasets
--------

Instructions on how to download datasets are available in each dataset file. You have to create a folder ``data/`` and put the raw data 
files inside.