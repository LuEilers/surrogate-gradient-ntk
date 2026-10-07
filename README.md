# A generalized neural tangent kernel for surrogate gradient learning
The code builds on the [Neural Tangents API](https://pypi.org/project/neural-tangents/), and the [Neural Tangents Cookbook](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/neural_tangents_cookbook.ipynb) was used as a reference.

## Requirements
The required dependencies can be installed as follows:
```
conda create -n sg_ntk python=3.10 -y
conda activate sg_ntk
pip install -r requirements.txt
```

## Figure 1
Data for Figure 1 is generated with [simulation_figure1.py](simulation_figure1.py). With this data, [plot_figure1.ipynb](plot_figure1.ipynb) is used to plot Figure 1 and Figure B.3.

## Figure 2
Data for Figure 2 is generated with [simulation_figure2.py](simulation_figure2.py). The simulation requires sufficiently large RAM. With this data, [plot_figure2.ipynb](plot_figure2.ipynb) is used to plot the data for Figure 2 and Figure B.4.

## Figure 3
Data for Figure 3 is generated with [simulation_figure3.py](simulation_figure3.py) for each hidden layer width $n=20, 100, 500$ separately. With this data, [plot_figure3.ipynb](plot_figure3.ipynb) is used to plot the data for Figure 3a and 3b.

## Base code
[ntk_definitions.py](ntk_definitions.py) contains the code for the simulation of empirical and analytic NTK.
[sg_ntk_definitions.py](sg_ntk_definitions.py) contains the modifications to the Neural Tangents package, the simulation of empirical and analytic SG-NTK, and code for SGL.

## Links
[Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/10f1737aa6347ccc555ac068e1b45523-Abstract-Conference.html)

[OpenReview](https://openreview.net/forum?id=kfdEXQu6MC)
