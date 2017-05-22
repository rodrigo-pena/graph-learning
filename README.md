# Graph Learning

This repository compiles some solutions for the problem of learning a graphical representation of a set of data points.

Currently, [learn-graph](https://github.com/rodrigo-pena/graph-learning/blob/master/learn_graph.py) implements algorithms from the following papers:

* Kalofolias, V. ["How to learn a graph from smooth signals"][kalofolias], AISTATS, 2016.
* Dong et al. ["Laplacian Matrix Learning for Smooth Graph Signal Representation"][dong]. ICASSP, 2015.
* Friedman, Hastie, and Tibshirani ["Sparse inverse covariance estimation with the graphical lasso"][glasso], Biostatistics 2008; 9 (3): 432-441.

The code is released under the terms of the [MIT license](LICENSE.txt).

[kalofolias]:  https://arxiv.org/abs/1601.02513
[glasso]: http://statweb.stanford.edu/~tibs/ftp/graph.pdf
[dong]: http://web.media.mit.edu/~xdong/paper/icassp2015.pdf

## Installation

1. Clone this repository.

   ```sh
   git clone https://github.com/rodrigo-pena/graph-learning
   cd graph-learning
   ```

2. Install the dependencies.
   
   ```sh
   pip install -r requirements.txt
   ```

3. Try the Jupyter notebooks.
   
   ```sh
   jupyter notebook
   ```

## Usage

To use this code on your data, you need:

1. An *n*-by-*m* data matrix of *n* variable observations in an *m*-dimensional space. Your graph will have *n* nodes.

See any of the jupyter notebooks for examples on how to call the methods using fabricated data.
Please get in touch if you are unsure about how to adapt the code to different settings.