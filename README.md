# TrueSkill Model
### Background 
This model is a variant of the TrueSkill model, a player ranking system for competitive games originally developed for Halo 2. It is a generalization of the Elo rating system in Chess. This model is based on one developed by Carl Rasmussen at Cambridge for his course on Machine Learning:  http://mlg.eng.cam.ac.uk/teaching/4f13/1920/

### Outline of the doc

1. Implementing the model
2. Examining the posterior for only two players and toy data
3. Stochastic Variational Inference on Two Players and Toy Data. 
4. Approximate Inference Conditioned on Tennis Data

* `Project.toml` packages for the Julia environment.
* `TrueSkill_src.jl` Julia code providing useful functions.
* `TrueSkill_starter.jl` starter code for assignment in Julia.
* `autograd_starter.py` some starter if you would like to use Python with autograd.
* `plots/` directory to store your plots.
* `tennis_data.mat` dataset containing outcomes of tennis games.
* `TrueSkill.pdf` concludes the model and the findings in each section. 
