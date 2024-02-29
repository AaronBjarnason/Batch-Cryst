#Cryst_model.py

# Defines the parameters for a 'typical experiment', and solves the crystallization process using the method of moments, 
# Used as part of 'batch_crystallizer_VSC.ipynb'

import pyomo.environ as pyo
from pyomo.dae import *
from pyomo.contrib.doe import (ModelOptionLib, DesignOfExperiments, MeasurementVariables, DesignVariables)
import numpy as np

def typical_experiment(beta=0.1/60):
  '''
  This function defines a typical experiment for batch crystallization

  Outputs:
  exp_conditions: returns dictionary of experimental conditions
  '''
  # Empty dictionary for experimental condtions
  exp_conditions = {}

  # Experimental parameters (Ci, Ti, tau, beta, eps, M_seed, Ls)

  exp_conditions['Ci'] = 116.68 * 1e-3      # initial concentration [g.cm-3]
  #exp_conditions['Ci'] = 112.68 * 1e-3 # different option

  exp_conditions['Ti'] = 323.9              # initial temperature [K]

  exp_conditions['tau'] = 7200      # process duration [s]

  # exp_conditions['beta'] = 0         
  exp_conditions['beta'] = beta             # cooling rate [K/s]

  exp_conditions['eps'] = 0.5 * 1e-3        # power density [W / g cryst]

#   exp_conditions['M_seed'] = 2.4 * 1e-3     # initial seed concentration [g.cm-3]
  exp_conditions['M_seed'] = 0              # initial seed concentration [kg.m-3]

  exp_conditions['Ls'] = 2e-5 * 1e3         # seed size [mm]

  return exp_conditions

# Creating the model
def Cryst_model(mod=None, model_option='parmest',exp_conditions=typical_experiment(),time_points=None):
    '''
    Create a Concrete Pyomo model for the crystallization problem

    We are debugging this model now. For parameter estimation and MBDoE,
    this function will be very helpful.

    Inputs:
    mod: model; set to None
    model_option: model option; choose between stage1, stage2, parmest
    exp_conditions: experimental conditions
    time_points: time points; set to None

    Outputs:
    m: Pyomo model
    '''

    ## Build the model ##
    # m = pyo.ConcreteModel()
    m=mod

    ## Model option ##
    model_option = ModelOptionLib(model_option)

    if model_option == ModelOptionLib.parmest:
        m = pyo.ConcreteModel()
        #print('created model')
        return_m = True
    elif model_option == ModelOptionLib.stage1 or model_option == ModelOptionLib.stage2:
        if not mod:
            raise ValueError(
                "If model option is stage1 or stage2, a created model needs to be provided."
            )
        return_m = False
    else:
        raise ValueError(
            "model_option needs to be defined as parmest, stage1, or stage2."
        )

    ## Fixed parameters ##

    # These parameters are not changing
    rho     = 1360 * 1e-6      # crystal density [g.mm-3]
    kv      = 0.523         # volume shape factor
    C1      = 2.209e-6 * 1e-3      # solubility parameter 1 (for g.cm-3)
    C2      = 5.419e-2      # solubility parameter 2

    # Establishing non-dimensional time parameter (i.e., t=tau*t_f)
    m.tau = exp_conditions['tau']

    ## Scaling parameters ##
    m.mu0_scale      = 1e5
    m.mu1_scale      = 1e4
    m.mu2_scale      = 1e2
    m.mu3_scale      = 1e1
    m.mu4_scale      = 1e0
    # m.B_scale        = 1e1
    # m.B_scale        = 1e2  # May change this back
    m.Bp_scale       = 1e1  # Primary nucl. scaling
    m.Bs_scale       = 1e2  # Seconday nucl. scaling
    m.G_scale        = 1e-5

    ## Fitting parameters ##
    # These are the parameters we are using for parameter estimation
    m.theta = {'kb1': 1e2, 'b1': 1, 'kb2': 1e9, 'b2': 2, 'kg': 2e-4, 'g': 1} # kb1 to unit 1/mL, kb2 does not have units changed due to balance on epsilon units, kg changed to mm/s. Unit conversions added

    # m.theta = {'kb1': 1e2, 'b1': 1, 'kb2': 1e3, 'b2': 2, 'kg': 2e-4, 'g': 1} # kb1 to unit 1/mL, kb2 does not have units changed due to balance on epsilon units, kg changed to mm/s. Unit conversions added

    ## Time ##
    m.t0 = pyo.Set(initialize=[0]) # Initial time
    if time_points is not None:
        # Do not specify time points
        # m.t = ContinuousSet(bounds=(0, 1)) # Time set
        m.t = ContinuousSet(bounds=(0, m.tau)) # Time set
    else:
        # m.t = ContinuousSet(bounds=(0, 1), initialize=time_points) # Time set
        m.t = ContinuousSet(bounds=(0, m.tau), initialize=time_points) # Time set

    ## Initial Conditions ##
    # Declare data used for only initial conditions

    c_sat = 1.01*C1*np.exp(C2*exp_conditions['Ti'])

    # Concentration
    #m.Ci = pyo.Param(m.t0, initialize = exp_conditions['Ci'], within = pyo.NonNegativeReals)
    m.Ci = pyo.Var(m.t0, initialize=c_sat, within=pyo.NonNegativeReals)

    # Temperature
    # m.Ti = pyo.Param(m.t0, initialize = exp_conditions['Ti'], within = pyo.NonNegativeReals)
    m.Ti = pyo.Var(m.t0, initialize = exp_conditions['Ti'], within = pyo.NonNegativeReals)

    # Epsilon
    m.eps = pyo.Param(m.t0, initialize = exp_conditions['eps'], within = pyo.NonNegativeReals)

    # M seed
    # m.M_seed = pyo.Param(m.t0, initialize = exp_conditions['M_seed'], within = pyo.NonNegativeReals)
    m.M_seed = pyo.Var(m.t0, initialize = exp_conditions['M_seed'], within = pyo.NonNegativeReals)

    # Ls
    m.Ls = pyo.Param(m.t0, initialize = exp_conditions['Ls'], within = pyo.NonNegativeReals)

    # Beta
    # For now, assume constant scaling
    m.beta = pyo.Var(m.t0, initialize = exp_conditions['beta'], within = pyo.NonNegativeReals)

    ## Declare Variables ##
    m.mu0 = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.mu0 = pyo.Var(m.t, initialize = exp_conditions['M_seed']/(rho*kv*exp_conditions['Ls']**3)/m.mu0_scale, within = pyo.NonNegativeReals)
    m.mu1 = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.mu1 = pyo.Var(m.t, initialize = exp_conditions['M_seed']/(rho*kv*exp_conditions['Ls']**4)/m.mu1_scale, within = pyo.NonNegativeReals)
    m.mu2 = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.mu2 = pyo.Var(m.t, initialize = exp_conditions['M_seed']/(rho*kv*exp_conditions['Ls']**5)/m.mu2_scale, within = pyo.NonNegativeReals)
    m.mu3 = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.mu3 = pyo.Var(m.t, initialize = exp_conditions['M_seed']/(rho*kv*exp_conditions['Ls']**6)/m.mu3_scale, within = pyo.NonNegativeReals)
    m.mu4 = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.mu4 = pyo.Var(m.t, initialize = exp_conditions['M_seed']/(rho*kv*exp_conditions['Ls']**7)/m.mu4_scale, within = pyo.NonNegativeReals)



    m.C = pyo.Var(m.t, initialize = exp_conditions['Ci'], within = pyo.NonNegativeReals)
    m.MT = pyo.Var(m.t, initialize = exp_conditions['M_seed'], within = pyo.NonNegativeReals)
    m.T = pyo.Var(m.t, initialize = exp_conditions['Ti'], within = pyo.NonNegativeReals)
    m.G = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.B = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    m.Bp = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    m.Bs = pyo.Var(m.t, initialize = 0, within = pyo.NonNegativeReals)
    # m.avg_vol = pyo.Var(m.t, initialize=1, within=pyo.NonNegativeReals)
    m.avg_vol = pyo.Var(m.t, initialize=exp_conditions['Ls'], within=pyo.NonNegativeReals)

    ## Declare Derivative Variable ##
    m.dmu0 = DerivativeVar(m.mu0)
    m.dmu1 = DerivativeVar(m.mu1)
    m.dmu2 = DerivativeVar(m.mu2)
    m.dmu3 = DerivativeVar(m.mu3)
    m.dmu4 = DerivativeVar(m.mu4)
    m.dC = DerivativeVar(m.C)
    m.dMT = DerivativeVar(m.MT)  # --> don't need a DAE, just an algebraic eqn (?)
    m.dT = DerivativeVar(m.T)


    '''
    m.kb1 = pyo.Param(initialize = m.theta['kb1'], mutable = True)
    m.b1 = pyo.Param(initialize = m.theta['b1'], mutable = True)
    m.kb2 = pyo.Param(initialize = m.theta['kb2'], mutable = True)
    m.b2 = pyo.Param(initialize = m.theta['b2'], mutable = True)
    m.kg = pyo.Param(initialize = m.theta['kg'], mutable = True)
    m.g = pyo.Param(initialize = m.theta['g'], mutable = True)
    '''
    ## Bounds ##
    low_b = 0.1
    high_b = 10

    ## New Bounds ##
    m.theta_lb = {'kb1': 1e1, 'b1': 1, 'kb2': 1e6, 'b2': 1, 'kg': 2e-6, 'g': 1}
    # m.theta_lb = {'kb1': 1, 'b1': 0.1, 'kb2': 1e4, 'b2': 0.1, 'kg': 2e-7, 'g': 1e-1}
    m.theta_ub = {'kb1': 1e4, 'b1': 4, 'kb2': 1e12, 'b2': 5, 'kg': 2e-2, 'g': 3}
    # m.theta_ub = {'kb1': 1e5, 'b1': 10, 'kb2': 1e13, 'b2': 10, 'kg': 2e-1, 'g': 5}

    # m.kb1 = pyo.Var(initialize = m.theta['kb1'], bounds=(low_b*m.theta['kb1'], high_b*m.theta['kb1']))
    # m.b1 = pyo.Var(initialize = m.theta['b1'], bounds=(low_b*m.theta['b1'], high_b*m.theta['b1']))
    # m.kb2 = pyo.Var(initialize = m.theta['kb2'], bounds=(low_b*m.theta['kb2'], high_b*m.theta['kb2']))
    # m.b2 = pyo.Var(initialize = m.theta['b2'], bounds=(low_b*m.theta['b2'], high_b*m.theta['b2']))
    # m.kg = pyo.Var(initialize = m.theta['kg'], bounds=(low_b*m.theta['kg'], high_b*m.theta['kg']))
    # m.g = pyo.Var(initialize = m.theta['g'], bounds=(low_b*m.theta['g'], high_b*m.theta['g']))

    # m.kb1 = pyo.Var(initialize = m.theta['kb1'], bounds=(m.theta_lb['kb1'], m.theta_ub['kb1']))
    m.b1 = pyo.Var(initialize = m.theta['b1'], bounds=(m.theta_lb['b1'], m.theta_ub['b1']))
    # m.kb2 = pyo.Var(initialize = m.theta['kb2'], bounds=(m.theta_lb['kb2'], m.theta_ub['kb2']))
    m.b2 = pyo.Var(initialize = m.theta['b2'], bounds=(m.theta_lb['b2'], m.theta_ub['b2']))
    # m.kg = pyo.Var(initialize = m.theta['kg'], bounds=(m.theta_lb['kg'], m.theta_ub['kg']))
    m.g = pyo.Var(initialize = m.theta['g'], bounds=(m.theta_lb['g'], m.theta_ub['g']))

    ## TRYING TO USE LOG TRANSFORMATION FOR K VALUES ##
    m.kb1 = pyo.Var(initialize = np.log(m.theta['kb1']), bounds=(np.log(m.theta_lb['kb1']), np.log(m.theta_ub['kb1'])))
    m.kb2 = pyo.Var(initialize = np.log(m.theta['kb2']), bounds=(np.log(m.theta_lb['kb2']), np.log(m.theta_ub['kb2'])))
    m.kg = pyo.Var(initialize = np.log(m.theta['kg']), bounds=(np.log(m.theta_lb['kg']), np.log(m.theta_ub['kg'])))

    ## Fixing the parameters ##
    m.kb1.fix()
    m.b1.fix()
    m.kb2.fix()
    m.b2.fix()
    m.kg.fix()
    m.g.fix()

    alt_rel_sat = True

    ## Functions and Constraints ##
    # Smoothing function for undersaturated
    # (1/2)(sqrt(z^2 + eps) + z)
    # For undersaturation, let z = -M
    # M + (-M) --> 0
    # For supersaturation, let z = M
    # M + M --> (1/2)*(2M) --> M
    # trying to max(0,z) and returns z if > 0
    def smooth_max(z):
        return 0.5*(pyo.sqrt(z**2+1e-8)+z)

    # Solubility expression
    def solubility(T):
        return C1*pyo.exp(C2*T)

    def inv_solubility(T):
        return pyo.exp(-C2*T)/C1

    # Nucleation and growth expressions
    # @m.Constraint(m.t)
    # def nucleation(m, t):
    #     # Relative supersaturation computation
    #     rel_sat = (m.C[t] - solubility(m.T[t]))/solubility(m.T[t])
    #     # Total nucleation rate
    #     #        B    =             Primary                 +                       Secondary
    #     return m.B[t] == ((m.kb1*smooth_max(rel_sat)**m.b1) + (m.kb2*smooth_max(rel_sat)**m.b2)*m.eps[0]*(m.MT[t])) / m.B_scale

    # Primary and Secondary nucleation separated (for scaling purposes)
    @m.Constraint(m.t)
    def prim_nucl(m, t):
        # Relative supersaturation computation
        if alt_rel_sat:
            rel_sat = m.C[t]*inv_solubility(m.T[t])-1
        else:
            rel_sat = (m.C[t] - solubility(m.T[t]))/solubility(m.T[t])

        # return m.Bp[t] == (m.kb1*smooth_max(rel_sat)**m.b1) / m.Bp_scale
        return m.Bp[t] == (pyo.exp(m.kb1)*smooth_max(rel_sat)**m.b1) / m.Bp_scale

    @m.Constraint(m.t)
    def sec_nucl(m, t):
        # Relative supersaturation computation
        if alt_rel_sat:
            rel_sat = m.C[t]*inv_solubility(m.T[t])-1
        else:
            rel_sat = (m.C[t] - solubility(m.T[t]))/solubility(m.T[t])

        # return m.Bs[t] == (m.kb2*smooth_max(rel_sat)**m.b2)*m.eps[0]*(m.MT[t]) / m.Bs_scale
        return m.Bs[t] == (pyo.exp(m.kb2)*smooth_max(rel_sat)**m.b2)*m.eps[0]*(smooth_max(m.MT[t])) / m.Bs_scale


    @m.Constraint(m.t)
    def growth(m, t):
        # Relative supersaturation computation
        if alt_rel_sat:
            rel_sat = m.C[t]*inv_solubility(m.T[t])-1
        else:
            rel_sat = (m.C[t] - solubility(m.T[t]))/solubility(m.T[t])
        # Growth rate expression
        # return m.G[t] == (m.kg*smooth_max(rel_sat)**m.g) / m.G_scale
        return m.G[t] == (pyo.exp(m.kg)*smooth_max(rel_sat)**m.g) / m.G_scale


    # Constraints for moments and concentration (fixed) for units of mm (L) and mL, or cm^3 (V)
    # @m.Constraint(m.t)
    # def ode1(m, t):
    #     return m.dmu0[t] == m.B[t] * m.B_scale / m.mu0_scale #* m.tau  # Units of # / cm^3 / s

    @m.Constraint(m.t)
    def ode1(m, t):
        return m.dmu0[t] == (m.Bp[t] * m.Bp_scale + m.Bs[t] * m.Bs_scale) / m.mu0_scale #* m.tau  # Units of # / cm^3 / s

    @m.Constraint(m.t)
    def ode2(m, t):
        return m.dmu1[t] == m.G[t] * m.G_scale * m.mu0[t] * m.mu0_scale / m.mu1_scale #* m.tau  # Units of mm * # / cm^3 / s

    @m.Constraint(m.t)
    def ode3(m, t):
        return m.dmu2[t] == 2 * m.G[t] * m.G_scale * m.mu1[t] * m.mu1_scale / m.mu2_scale #* m.tau  # Units of mm^2 * # / cm^3 / s

    @m.Constraint(m.t)
    def ode4(m, t):
        return m.dmu3[t] == 3 * m.G[t] * m.G_scale * m.mu2[t] * m.mu2_scale / m.mu3_scale #* m.tau  # Units of mm^3 * # / cm^3 / s

    @m.Constraint(m.t)
    def ode5(m, t):
        return m.dmu4[t] == 4 * m.G[t] * m.G_scale * m.mu3[t] * m.mu3_scale / m.mu4_scale #* m.tau  # Units of mm^4 * # / cm^3 / s

    @m.Constraint(m.t)
    def ode6(m, t):
        return m.dC[t] == -3 * kv * rho * m.G[t] * m.G_scale * smooth_max(m.mu2[t]) * m.mu2_scale #* m.tau  # Units of g / cm^3 / s

    # @m.Constraint(m.t)
    # def MT_balance(m, t):
    #     return m.MT[t] == rho * kv * m.mu3[t] * m.mu3_scale # Units of grams.mm-3

    @m.Constraint(m.t)
    def ode7(m, t):
        return m.dMT[t] == 3 * kv * rho * m.G[t] * m.G_scale * smooth_max(m.mu2[t]) * m.mu2_scale # Units of grams.mm-3

    @m.Constraint(m.t)
    def ode8(m, t):
        return m.dT[t] == -m.beta[0] # * m.tau
        # beta[0] is 0 (temperature decay), if we set it to something else we will get something else
        # change beta 0 to something physical
        # (0,(0.5/60))

    @m.Constraint(m.t)
    def average_volume(m,t):
        return (m.mu4[t] + 1e-8) * m.mu4_scale == m.avg_vol[t] * (m.mu3[t] + 1e-8) * m.mu3_scale  # / 1000    Multiplying by 1000 to get in micrometers????
        # return (m.mu1[t] + 1e-8) * m.mu1_scale == m.avg_vol[t] * (m.mu0[t] + 1e-8) * m.mu0_scale


    ## Initial conditions (fixed), in units desired (mm and cm^3) ##
    m.mu0[0.0].fix(m.M_seed[0]/(rho*kv*m.Ls[0]**3)/m.mu0_scale)
    m.mu1[0.0].fix(m.mu0[0.0]*m.mu0_scale*m.Ls[0]/(m.mu1_scale))
    m.mu2[0.0].fix(m.mu0[0.0]*m.mu0_scale*(m.Ls[0])**2/m.mu2_scale)
    m.mu3[0.0].fix(m.mu0[0.0]*m.mu0_scale*(m.Ls[0])**3/m.mu3_scale)
    m.mu4[0.0].fix(m.mu0[0.0]*m.mu0_scale*(m.Ls[0])**4/m.mu4_scale)

    m.C[0.0].fix(m.Ci[0])
    m.MT[0.0].fix(m.M_seed[0])
    m.T[0].fix(m.Ti[0])

    # If concentration is undersaturated, these expressions should be 0. May not have to initialize these values.
    # m.Bp[0.0].value = (np.exp(m.kb1.value)* ((((m.C[0.0].value - C1*np.exp(C2*m.T[0.0].value)) / (C1*np.exp(C2*m.T[0.0].value))))**m.b1.value)) / m.Bp_scale
    # m.G[0.0].value = (np.exp(m.kg.value)*((((m.C[0.0].value - C1*np.exp(C2*m.T[0.0].value)) / (C1*np.exp(C2*m.T[0.0].value)))))**m.g.value) / m.G_scale
    # m.Bs[0.0].value = (np.exp(m.kb2.value) * m.eps[0.0] * m.M_seed[0.0] * ((((m.C[0.0].value - C1*np.exp(C2*m.T[0.0].value)) / (C1*np.exp(C2*m.T[0.0].value)))))**m.b2.value) / m.Bs_scale

    ## Now fix the cooling rate ##
    m.beta[0].fix(exp_conditions['beta'])


    if return_m:
      return m


exp = typical_experiment()

# my_timepoints = np.array([0.00000000e+00, 2.20433422e-11, 2.20455465e-07, 2.42478969e-06,
#         1.10338916e-05, 3.07350790e-05, 6.84677150e-05, 1.36272759e-04,
#         2.55316549e-04, 4.62543213e-04, 8.22160101e-04, 1.44559636e-03,
#         2.52555626e-03, 4.35738178e-03, 6.18920730e-03, 8.02103282e-03,
#         1.18165311e-02, 1.56120294e-02, 1.94075277e-02, 2.32030259e-02,
#         2.94074899e-02, 4.07352989e-02, 6.17812187e-02, 7.45454635e-02,
#         8.73097084e-02, 1.00073953e-01, 1.12838198e-01, 1.32554654e-01,
#         1.69522761e-01, 2.06490868e-01, 2.43458975e-01, 2.80427082e-01,
#         3.17395189e-01, 3.54363295e-01, 3.91331402e-01, 4.28299509e-01,
#         4.65267616e-01, 5.22967872e-01, 7.18796237e-01, 8.43719621e-01,
#         9.68643004e-01, 1.09356639e+00, 1.32109920e+00, 1.54863202e+00,
#         2.05517540e+00, 2.41828785e+00, 2.78140031e+00, 3.14451276e+00,
#         3.72198774e+00, 4.29946273e+00, 4.87693771e+00, 5.45441270e+00,
#         6.69464215e+00, 8.57792195e+00, 1.04612018e+01, 1.23444816e+01,
#         1.42277614e+01, 1.79566619e+01, 2.44477706e+01, 3.50091373e+01,
#         4.55705041e+01, 5.61318708e+01, 6.66932376e+01, 7.72546043e+01,
#         9.53412821e+01, 1.13427960e+02, 1.31514638e+02, 1.60093793e+02,
#         1.88672949e+02, 2.31647593e+02, 2.74622237e+02, 3.17596881e+02,
#         3.87783749e+02, 4.57970616e+02, 5.28157484e+02, 5.98344352e+02,
#         6.68531219e+02, 7.38718087e+02, 8.08904955e+02, 8.79091822e+02,
#         9.49278690e+02, 1.01946556e+03, 1.08965243e+03, 1.19557918e+03,
#         1.30150594e+03, 1.40743270e+03, 1.51335946e+03, 1.61928622e+03,
#         1.72521298e+03, 1.83113974e+03, 1.93706650e+03, 2.04299325e+03,
#         2.14892001e+03, 2.25484677e+03, 2.36077353e+03, 2.46670029e+03,
#         2.57262705e+03, 2.67855381e+03, 2.78448057e+03, 2.89040732e+03,
#         2.99633408e+03, 3.15737308e+03, 3.31841207e+03, 3.47945107e+03,
#         3.64049006e+03, 3.80152906e+03, 3.96256805e+03, 4.12360705e+03,
#         4.28464604e+03, 4.44568504e+03, 4.60672403e+03, 4.76776303e+03,
#         4.92880202e+03, 5.08984102e+03, 5.25088001e+03, 5.41191901e+03,
#         5.57295800e+03, 5.73399700e+03, 5.89503599e+03, 6.05607499e+03,
#         6.21711398e+03, 6.46365041e+03, 6.71018683e+03, 6.95672326e+03,
#         7.20325969e+03, 7.44979611e+03, 7.69633254e+03, 7.94286897e+03,
#         8.18940539e+03, 8.43594182e+03, 8.68247824e+03, 8.92901467e+03,
#         9.17555110e+03, 9.42208752e+03, 9.66862395e+03, 9.91516037e+03,
#         1.01616968e+04, 1.04082332e+04, 1.06547697e+04, 1.09013061e+04,
#         1.11478425e+04, 1.13943789e+04, 1.16409154e+04, 1.18874518e+04,
#         1.22686332e+04, 1.26498146e+04, 1.30309959e+04, 1.34121773e+04,
#         1.37933587e+04, 1.41745401e+04, 1.47752966e+04, 1.53760531e+04,
#         1.59768096e+04, 1.65775660e+04, 1.71783225e+04, 1.77790790e+04,
#         1.83798355e+04, 1.89805920e+04, 1.95813485e+04, 2.01821050e+04,
#         2.07828615e+04, 2.13836180e+04, 2.16000000e+04])

# m = Cryst_model(exp_conditions=exp, time_points=my_timepoints)