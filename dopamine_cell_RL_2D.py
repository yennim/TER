# ==============================================================================
"""Dopamine cell prediction activity"""
# ==============================================================================
__author__ = "Anais GRIMAL, Hoang-Yen PHAM"
__date__ = "03/04/2017"
# ==============================================================================

from bokeh.core.properties import Any, Dict, Instance, String
from bokeh.models import ColumnDataSource, Div, Column,  LayoutDOM
from bokeh.plotting import figure, output_file, show
from bokeh.io import show

import numpy as np

import model

# ==============================================================================
n_trials = 100

T = 5
N = 23 # time steps occurrence
γ = 0.98 # discount factor
α = 0.005 # learning rate
λ = 0.9 # eligibility trace parameter


# running the model
tdmodel = model.TDModel(λ=λ, α=α, γ=γ, N=N, T=T)
for _ in range(n_trials):
    tdmodel.trial()
tdmodel.trial(True, False)
            
# plotting
X = np.arange(0,T,T/N)
p = figure(title="TD(0.9) Model", x_axis_label="Time steps",
           y_axis_label="Prediction error", y_range=[-0.1,1.4],
           plot_width=400, plot_height=300)
p.line(X, tdmodel.δ_history[-2], legend="Cued reward", line_color="black", line_width=2)
p.line(X, tdmodel.δ_history[-1], legend="Omit cue 2", line_color="red", line_dash="dotted", line_width=2)
##p.circle(20*T/N, 1, size=4, line_color="red", fill_color="red")
##p.circle(5*T/N, 1, size=4, line_color="#56BA1B", fill_color="#56BA1B")
##p.circle(15*T/N, 1, size=4, line_color="#56BA1B", fill_color="#56BA1B")

show(p)
            
