# Simple Flow Matching
A minimal demo of Conditional Flow Matching with Optimal Transport adapted from the 100 lines of code implementation [here](https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa).

The example notebook walks through the training and inference workflow [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anoushka2000/simple-flow-matching/blob/main/simple_flow_matching/train.ipynb)


The [Torch-CFM library](https://github.com/atong01/conditional-flow-matching) provides scalable and flexible support for training CNFs.


## Reference Paper

The algorithms implemented and notation used in the notes and code follows:
<details>
<summary>
Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. Flow Matching for Generative Modeling, 2023.
</summary>

```bibtex
@article{lipman2023flowmatchinggenerativemodeling,
      title={Flow Matching for Generative Modeling}, 
      author={Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le},
      year={2023},
      eprint={2210.02747},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2210.02747}, 
}
```
</details>



## Installation
Follow the steps given below to install `simple-flow-matching` in a `conda` virtual environment.

```bash
# Clone this repository to your machine
git clone https://github.com/anoushka2000/simple-flow-matching.git

# On artemis
module purge
module --ignore_cache load python/3.11.5 cuda/12.1.1

# Create a virtual environment
conda create --name flow_matching python=3.11

# Activate the environment
conda activate flow_matching

# Install package in the environment
cd simple-flow-matching
pip install .
```
