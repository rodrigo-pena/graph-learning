# Graph Learning

This repository compiles some solutions for the problem of learning a graphical representation of a set of data points.

Currently, algorithms from the following papers are implemented:

* Kalofolias, V. [How to learn a graph from smooth signals][kalofolias], AISTATS, 2016.

The code is released under the terms of the [MIT license](LICENSE.txt).

[kalofolias]:  https://arxiv.org/abs/1601.02513

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