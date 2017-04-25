from model import *
import numpy as np

def test1():
    """Test for λ=0 and 1 stimulus"""
    tdmodel = TDModel(0, 0.05, 0.98, 4, 5, [1])
    for trials in range(2):
        eligibility = tdmodel.trial([True, False], 3)

    assert tdmodel.δ_history[-1][-1] == 1-tdmodel.α
    assert tdmodel.w.all() == np.array([(tdmodel.α**2)*tdmodel.γ, (2*tdmodel.α)-(tdmodel.α**2), 0, 0]).all()
    assert eligibility.all() == np.array([0, 1, 0, 0]).all()   

def test2():
    """Test for 2 stimuli"""
    tdmodel = TDModel(1, 0.05, 0.98, 5, 5, [0,2])
    for trials in range(2):
        eligibility = tdmodel.trial([True, True], 4)

    assert tdmodel.δ_history[-1][-4] == (tdmodel.γ-1)*tdmodel.α # for [-1][-4]
##    assert tdmodel.w[0].all() == np.array([tdmodel.α*(1+2*tdmodel.α*tdmodel.γ-tdmodel.α),
##                                           tdmodel.α*(1+tdmodel.α*tdmodel.γ-tdmodel.α),
##                                           tdmodel.α, tdmodel.α, tdmodel.α]).all()
##    assert tdmodel.w[1].all() == np.array([0, 0, 0, 0, 0]).all()
##    assert eligibility[0].all() == np.array([1, 1, 0, 0, 0]).all()
##    assert eligibility[1].all() == np.array([0, 0, 0, 0, 0]).all()
