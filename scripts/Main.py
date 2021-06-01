######################################################
#  Main file running the code of: 
##################
#  A toolkit for solving overlapping generations models with 
#  family-linked bequest and intergenerational skill transmission
##################

####
# Structure:
#   Part 1: Preambles
#   Part 2: Define parameters
#   Part 3: Solve full model
#       - Baseline calibration with estimated parameters
#   Part 4: Simulation of the full model
#   Part 5: Estimation (outcommented)
#   Part 6: Plots and tables of full model
#   Part 7: Simulation of the model with random skill transmission
#   Part 8: Solve and simulate model with no bequest motive

########################################
# Part 1: Perambles

import numpy as np
import tools as tools
import model as model
import matplotlib.pyplot as plt # needed if Part 5 Estimation uncommented
import scipy.optimize as optimize # needed if Part 5 Estimation uncommented
import scipy.stats as stats
np.set_printoptions(suppress=True)

# define your directory for figures and tex-files for tables
save_to = 'C:/Users/tima.eco/CBS - Copenhagen Business School/Frederik Bjørn Christensen - TermPaper/Github/figtabs/'

########################################
# Part 2: Define parameters

par = dict()
###
# a) fixed parameters
par['T'] = 6
par['r'] = 4  # Number of periods in the workforce
par['ch'] = 4 # Age when children are 1
par['o'] = 1 # scale parameter, set to 1 and thus not mentioned in the paper

###
# b) adjustable economic parameters
par['β'] = 0.85 # patience
par['R'] = 1/par['β'] # exogenous interest rate
par['w'] = 1 # wage rate
par['ρ'] = 2 # Inverse elasticity of intertemporal substitution of consumption
par['γ'] = 2 # Inverse elasticity of intertemporal substitution of warm-glow bequest
par['a̲'] = 12.49087704 #12.38 # bequest shifter
par['κ'] = 4.68242024 #4.74 # strength of bequest motive
par['σ_eps'] = np.sqrt(0.3) # std of epsilon / normal shock in the AR1 // variance = 0.3
par['α'] = 0.85 # persistence in productivity shock / AR1 paramenter
par['σ_z'] = np.sqrt(par['σ_eps']**2/(1-par['α']**2)) # unconditional std of productivity
par['S'] = 7 # number of Gauss-Hermite nodes
par['ψ'] = np.concatenate([np.ones([par['r']]),np.array([0.83,0.58])*np.ones([par['T']-par['r']]),np.zeros([par['ch']])]) # Mortality pattern course SSA
par['l'] = np.concatenate([np.array([48.6,74.3,77.8,63.6])/np.mean(np.array([48.6,74.3,77.8,63.6])),np.zeros([par['T']-par['r']+1])]) # Age dependend-prod pattern (CBO)
quintiles_income = np.array([350,480,659,917,2149])  # income by Quintiles USA 2016 Source: CBO, The distribution of household income, 2016
par['q'] = np.repeat(0.2,5) # quintile populaiton shares
par['H'] = quintiles_income/np.sum(quintiles_income)/par['q'] # human captial / permanent skill/income heterogeneity
par['P_markov'] = np.array([[0.337,0.242,0.178,0.134,0.109],[0.28,0.242,0.198,0.16,0.12],[0.184,0.217,0.221,0.208,0.17],[0.124,0.176,0.22,0.244,0.236],[0.075,0.123,0.183,0.254,0.365]])

###
# c) adjustable computational parameters
par['M_max'] = par['H'][-1]*par['w']*par['r']*np.mean(par['l'][0:par['r']])*stats.lognorm.ppf(0.99,par['σ_z']) # Endogenous max of cash-on-hand grid
par['gridsize_m'] = 100 # Cash-on-hand gridsize in the stochastic part of the model
par['gridsize_deterministic_m'] = 100 # Cash-on-hand gridsize in the deterministic part of the model
par['gridsize_z'] = 20 # Gridsize of labour productivity process
par['𝒢_m_det'] = tools.nonlinspace(10e-6,par['M_max'],par['gridsize_deterministic_m']-1,1) # Grid of cash-on-hand in the final period
par['𝒢_a_det'] = tools.nonlinspace(10e-6,par['M_max'],par['gridsize_deterministic_m']-1,1) # for z we choose to only extrapolate for 1%  // np.sqrt(par['σ']^2/(1-par['α']^2) = unconditional std of z 
par['𝒢_z'] = tools.nonlinspace(-2.576*par['σ_z'],2.576*par['σ_z'],par['gridsize_z'],1) # 99%-confidenceband around 

########################################
# Part 3: Solve full model
Cstar_det,𝒢_M_det,H_Cstar,H_𝒢_MEn = model.solve(par)


########################################
# Part 4: Simulation of the full model

N = 100000 # number of simulated agents
b_iter = 10 # iterations over simulated generations in order for the wealth distribution to converge
simY,simZ,simM,simC,simA,simMean,simStd,hi_children,DOA = model.sim(𝒢_M_det,Cstar_det,H_𝒢_MEn,H_Cstar,par,N,b_iter)


########################################
# Part 5: Estimation
# # ---- (κ,a̲) for full model - It is outcommented as a default. The solution to the estimation (κ,a̲)= (4.68242024, 12.49087704) is used above

# uncomment here
# θ = [10,10] # [κ,a̲]
# targets = [0.62,0.163] # [gini at retirement,P60P80]
# linear_constraint = optimize.LinearConstraint([[1,0],[0,1]],[0,0],[100,100]) # linear constraints
# solution = optimize.minimize(model.objective, θ,args=(targets,par,N,b_iter),constraints=linear_constraint,method='SLSQP',tol = 1e-10,options={"disp":True,"maxiter": 100, 'ftol': 1e-08})
# params = solution.x
# par['κ'],par['a̲'] = params[0],params[1] # the solution is (κ,a̲)= (4.68242024, 12.49087704)


########################################
# Part 6: Plots and tables of full model


# a) Export targeted moments: Gini at retirement and wealth shares of different percentile ranges (tex table output)
gini,P60P80 = model.Moments(simM,simA,DOA,par)
with open(save_to+ "gini.tex", "w") as txt_gini:
        txt_gini.write("Wealth Gini at retirement & 0.62 &" +  format(gini, '.2f')  +"\\\\"  )
with open(save_to+ "P60P80.tex", "w") as txt_P60P80:
        txt_P60P80.write("Wealth share of 60th-80th Percentile & 0.163 &" +  format(P60P80, '.3f')  +"\\\\"  )


# b) Print gini and wealth shares of different percentile ranges 
WealthDistTable = np.around([100*x for x in model.Table(par,simM,N)],1)
print('Gini at retirement: ' +  format(gini, '.2f'))
print('Percentile 0-40: ' +  str(WealthDistTable[0]))
print('Percentile 40-60: ' +  str(WealthDistTable[1]))
print('Percentile 60-80: ' +  str(WealthDistTable[2]))
print('Percentile 80-90: ' +  str(WealthDistTable[3]))
print('Percentile 90-95: ' +  str(WealthDistTable[4]))
print('Percentile 95-99: ' +  str(WealthDistTable[5]))
print('Percentile 99-100: ' +  str(WealthDistTable[6]))

# c) Export wealth shares of different percentile ranges (tex table output)
with open(save_to+ "baseline.tex", "w") as txt_baseline:
        txt_baseline.write("Full model & " +  format(gini, '.2f') + "&" + str(WealthDistTable[0])+ "&" + str(WealthDistTable[1])+ "&" + str(WealthDistTable[2])+ "&" + str(WealthDistTable[3])+ "&" + str(WealthDistTable[4])+ "&" + str(WealthDistTable[5])+ "&" + str(WealthDistTable[6]) +"\\\\"  )

# d) Plot and export and figures of full model
model.plots(par,simC,simM,simA,simMean,simStd,hi_children,save_to)

########################################
# Part 7: Simulation of the model with random skill transmission

# create random Markov skill transition matrix
rowmarkov = np.repeat(0.2,5)
par['P_markov'] = np.array([rowmarkov,rowmarkov,rowmarkov,rowmarkov,rowmarkov])
# simulate
simY,simZ,simM,simC,simA,simMean,simStd,hi_children,DOA = model.sim(𝒢_M_det,Cstar_det,H_𝒢_MEn,H_Cstar,par,N,b_iter)
# Print and export gini and wealth shares of different percentile ranges 
gini,P60P80 = model.Moments(simM,simA,DOA,par)
WealthDistTable = np.around([100*x for x in model.Table(par,simM,N)],1)
print('Gini at retirement: ' +  format(gini, '.2f'))
print('Percentile 0-40: ' +  str(WealthDistTable[0]))
print('Percentile 40-60: ' +  str(WealthDistTable[1]))
print('Percentile 60-80: ' +  str(WealthDistTable[2]))
print('Percentile 80-90: ' +  str(WealthDistTable[3]))
print('Percentile 90-95: ' +  str(WealthDistTable[4]))
print('Percentile 95-99: ' +  str(WealthDistTable[5]))
print('Percentile 99-100: ' +  str(WealthDistTable[6]))
with open(save_to+ "randomskill.tex", "w") as txt_random:
        txt_random.write("Random skill transmission & " +  format(gini, '.2f') + "&" + str(WealthDistTable[0])+ "&" + str(WealthDistTable[1])+ "&" + str(WealthDistTable[2])+ "&" + str(WealthDistTable[3])+ "&" + str(WealthDistTable[4])+ "&" + str(WealthDistTable[5])+ "&" + str(WealthDistTable[6]) +"\\\\"  )


########################################
# Part 8: Solve and simulate model with no bequest motive

# use initial skill transmission markov matrix
par['P_markov'] = np.array([[0.337,0.242,0.178,0.134,0.109],[0.28,0.242,0.198,0.16,0.12],[0.184,0.217,0.221,0.208,0.17],[0.124,0.176,0.22,0.244,0.236],[0.075,0.123,0.183,0.254,0.365]])
# no bequest motive
par['κ'] = 0 #4.74 # strength of bequest motive
# Solve the model
Cstar_det,𝒢_M_det,H_Cstar,H_𝒢_MEn = model.solve(par) 
# Simulate the model
simY,simZ,simM,simC,simA,simMean,simStd,hi_children,DOA = model.sim(𝒢_M_det,Cstar_det,H_𝒢_MEn,H_Cstar,par,N,b_iter)
# Print and export gini and wealth shares of different percentile ranges 
gini,P60P80 = model.Moments(simM,simA,DOA,par)
WealthDistTable = np.around([100*x for x in model.Table(par,simM,N)],1)
print('Gini at retirement: ' +  format(gini, '.2f'))
print('Percentile 0-40: ' +  str(WealthDistTable[0]))
print('Percentile 40-60: ' +  str(WealthDistTable[1]))
print('Percentile 60-80: ' +  str(WealthDistTable[2]))
print('Percentile 80-90: ' +  str(WealthDistTable[3]))
print('Percentile 90-95: ' +  str(WealthDistTable[4]))
print('Percentile 95-99: ' +  str(WealthDistTable[5]))
print('Percentile 99-100: ' +  str(WealthDistTable[6]))
with open(save_to+ "nobequest.tex", "w") as txt_nobequest:
        txt_nobequest.write("No bequest motive & " +  format(gini, '.2f') + "&" + str(WealthDistTable[0])+ "&" + str(WealthDistTable[1])+ "&" + str(WealthDistTable[2])+ "&" + str(WealthDistTable[3])+ "&" + str(WealthDistTable[4])+ "&" + str(WealthDistTable[5])+ "&" + str(WealthDistTable[6]) +"\\\\"  )

