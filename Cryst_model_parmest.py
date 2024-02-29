# Cryst_model_parmest

# Accepts the simulated results from previous experiments (recorded in 'data') to estimate kinetic parameter values for the crystallization system
# Used as part of 'batch_crystallizer_VSC.ipynb'

import pyomo.environ as pyo
from pyomo.dae import *
from pyomo.contrib.doe import (ModelOptionLib, DesignOfExperiments, MeasurementVariables, DesignVariables)
import numpy as np
from Cryst_model import *

def typical_experiment(beta=0.1/60):
  '''
  This function defines a typical experiment for batch crystallization

  Outputs:
  exp_conditions: returns dictionary of experimental conditions
  '''
  # Empty dictionary for experimental condtions
  exp_conditions = {}

  # Experimental parameters (Ci, Ti, tau, beta, eps, M_seed, Ls)

  exp_conditions['Ci'] = 116.68 * 1e-3  # initial concentration [g.cm-3]
  #exp_conditions['Ci'] = 112.68 * 1e-3 # different option

  exp_conditions['Ti'] = 323.9          # initial temperature [K]

  exp_conditions['tau'] = 7200          # process duration [s]

  # exp_conditions['beta'] = 0         
  exp_conditions['beta'] = beta         # cooling rate [K/s]

  exp_conditions['eps'] = 0.5 * 1e-3    # power density [W / g cryst]

#   exp_conditions['M_seed'] = 2.4 * 1e-3     # initial seed concentration [g.cm-3]
  exp_conditions['M_seed'] = 0          # initial seed concentration [kg.m-3], [g.cm-3]

  exp_conditions['Ls'] = 2e-5 * 1e3     # seed size [mm]

  return exp_conditions


def Cryst_model_parmest(data):
    '''
    Special create_model function for parmest
    '''
    ## Setup experiment model
    exp = typical_experiment()
    exp['Ci'] = data.Ci.iloc[0]
    exp['Ti'] = data.Ti.iloc[0]

    m = Cryst_model(exp_conditions=exp,time_points=data.time.to_list())

    ## Numerical intergrate to initialize

    # Solve using simulator
    sim = Simulator(m, package='casadi')
    tsim, profiles = sim.simulate(integrator = 'idas')

    ## Discretize
    pyo.TransformationFactory('dae.finite_difference').apply_to(m, nfe=100,scheme='BACKWARD')
    sim.initialize_model()
    ## Prepare data

    C_measure={}
    MT_measure={}
    avg_vol_measure={}
    t_measure = data.time.to_list()
    for i,t in enumerate(data.time):
        C_measure[float(t)] = data.Ci[i]
        MT_measure[float(t)] = data.MT[i]
        avg_vol_measure[float(t)] = data.avg_volume[i]

    # Least squares objective

    def ComputeFirstStageCost_rule(m):
        return 0
    m.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(m):
      # measurements
        return sum((m.C[t] - C_measure[t]) ** 2 + (m.MT[t] - MT_measure[t]) ** 2
                       + (m.avg_vol[t] - avg_vol_measure[t]) ** 2 for t in t_measure)
        # return sum((m.C[t] - C_measure[t]) ** 2
                      #  + (100 * m.avg_vol[t] - 100 * avg_vol_measure[t]) ** 2 for t in t_measure)
    m.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    # return the sum of the first-stage and second-stage costs as the objective function
    def total_cost_rule(m):
        return m.FirstStageCost + m.SecondStageCost

    m.total_cost_objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return m