---
title: Set Up Python Environment for CogSci 131
date: 2022-01-19
math: true
tags:
    - python
    - jupyter
    - cogsci
categories:
- tutorials
keywords:
    - python
    - jupyter 
    - cogsci
include_toc: true

---

# Option #1: Conda Virtual Environment

To avoid nightmares down the line👇, use virtual environments for your local stuff. 

{{< figure src="https://imgs.xkcd.com/comics/python_environment_2x.png" width="400">}}

1. **Install Miniconda**: Install the right version for your operation system from [this page](https://docs.conda.io/en/latest/miniconda.html) (I used [Anaconda](https://www.anaconda.com/products/individual) years ago that comes with doznes of pre-installed packages, which is more cumbersome or convenient, depending on the perspective)

{{< figure src="https://www.dropbox.com/s/pm4p2v4n57adf5k/miniconda.png?raw=1" width="600" caption="Selection of latest Miniconda installers (if your machine has an Apple silicon chips, pick the third option under MacOS).">}}

2. **Create environment**: Open a terminal 👉 `cd` to where you wanna launch Jupyter (for me it's `cd /Users/apple/Documents/spring2022/cogsci131/notebooks`) 👉 run this command `conda create --name cogsci131 python=3.9.5`
    - "cogsci131" is the name of the environment; you can use whatever you like
    - Python 3.9.5 is the lasted version that conda installs as of writing; replace with the latest version when you do this 

3. **Activate environment**: In the same directory, run `conda activate cogsci131`

4. **Install packages**: For this course (or cognitive science in general), you most likely need to `conda install pandas numpy scipy matplotlib seaborn jupyterlab`
    - After all the installations are completed, you should be able to launch [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) from this environment: `jupyter lab`
    {{< figure src="https://www.dropbox.com/s/i83bax801n4lnmm/jupyterlab.png?raw=1" width="500" caption="Jupyter Lab opens a web browser 👉 you can start a kernel and create a notebook from there" >}}
5. (Optional) **Export environment**: If for some reason you need to recreate the above environment (nuked it or changed computers), you can export the settings to a [YAML](https://en.wikipedia.org/wiki/YAML) file (`conda env export > cogsci131.yml`) and create a new environment from the saved file (`conda create --name cogsci131 --file cogsci131.yml`)

# Option #2: Google Colab

If you wanna avoid the pain for now, you can use [Google Colab](https://colab.research.google.com/), which is like Google Docs, but for Jupyter notebooks. The user interface is slightly different between the two. Colab makes it easy to share your work, but if you wanna install a package permanently on Colab, you may need to link it to Google Drive (here's [how](https://stackoverflow.com/questions/55253498/how-do-i-install-a-library-permanently-in-colab)).

# Further Resources

- [**The Good Research Code Handbook**](https://goodresearch.dev/): Having a Python environment is sufficient for this course but is only the starting point of scientific computing. This book teaches you how to organize your computational projects from beginning to end.
- **Bayesian Modeling and Computation in Python** ([online version](https://bayesiancomputationbook.com/welcome.html), [physical copy](https://www.routledge.com/Bayesian-Modeling-and-Computation-in-Python/Martin-Kumar-Lao/p/book/9780367894368)): Exact Bayesian inference is expensive --- both for us as computational modelers and for the mind to carry out computations in everyday reasoning. For Python folks, PyMC3 is perhaps the most popular package for approximate Bayesian inference using [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) or [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VI) and this new book from the PyMC developers is a great way to learn it.
- [**How to Cog Sci**](https://vimeo.com/showcase/howtocogsci): Prof. Todd Gureckis at NYU has a shockingly well-produced video series introducing Jupyter, Python, stats, and data wrangling (the production quality is definitely up there the best professional YouTubers). It's amazing to imagine how Todd patiently teaches the basics extremely well while being an incredibly prolific scientist and open-source contributor...