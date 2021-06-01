import numpy as np
import scipy.optimize as optimize
import quantecon as qe
import matplotlib.pyplot as plt
import scipy.stats as stats
import tools


def solveT(par): # solve static last period consumption choice 
    β = par['β']
    ρ = par['ρ']
    γ = par['γ']
    κ = par['κ']
    o = par['o']
    a̲ = par['a̲']
    𝒢_m_det = par['𝒢_m_det']
    
    solvec = np.empty(𝒢_m_det.size) # solution storage

    for im,m in enumerate(𝒢_m_det):
        obj_fun = lambda c: ((c/o)**(-ρ) - β*κ*((m-c+a̲)/o)**(-γ))**2
        res = optimize.minimize_scalar(obj_fun,bounds=[0,m+1.0e-4],method='bounded',options={'xatol': 1e-4, 'maxiter': 10000})
        solvec[im] = res.x
    return solvec


def f_gini(InputVector,strat): # Calculation of Gini for a given wealth vector
    if len(InputVector)<strat:
        print('Input must have length greater than strat')    
        return True
    
    # InputVector May not be divisible by strat
    Dim = int(strat*np.floor(len(InputVector)/strat))
    New = np.sort(np.random.choice(InputVector,Dim,replace=False))
    Reshaped = np.reshape(New,[strat,int(Dim/strat)]).T    
    ColSum = np.sum(Reshaped,axis=0)
    Perc = ColSum/sum(ColSum)

    # Preallocating
    GiniSum = np.zeros([strat,strat])

    for l in range(strat):
        for j in range(strat):
            GiniSum[l,j] = abs(Perc[l] - Perc[j]);

    return sum(sum(GiniSum))/(2*strat)

def solve(par): # Solves the model
    # Unpacking Parameters
    β = par['β']
    γ = par ['γ']
    a̲ = par['a̲']
    κ = par['κ']
    σ_z = par['σ_z']
    ψ = par['ψ']
    T = par['T']
    ch = par['ch']
    o = par['o']
    R = par['R']
    ρ = par['ρ']
    S = par['S']
    l = par['l']
    H = par['H']
    w = par['w']
    α = par['α']
    gridsize_z = par['gridsize_z']
    gridsize_m = par['gridsize_m']
    gridsize_deterministic_m = par['gridsize_deterministic_m']
    𝒢_m_det = par['𝒢_m_det']
    𝒢_a_det = par['𝒢_a_det']
    𝒢_z = par['𝒢_z']
    H = par['H']

    
    # Preallocating
    H_Cstar = [] # This list will consist of Cstar for different levels of h
    H_𝒢_MEn = [] # This list will consist of Cash-on-hand grids for different levels of h
    Cstar_det = np.zeros([T,gridsize_deterministic_m]) # Cstar for deterministic periods // Note, dim1=6, despite only 3 rows filled
    𝒢_M_det = np.zeros([T,gridsize_deterministic_m])   # Endogenous grid for cash-on-hand in deterministic periods // Note, dim1=6, despite only 3 rows filled
    
    # Load Gauss-Hermite weights and nodes
    x,wi = tools.gauss_hermite(S)
    ω = wi/np.sqrt(np.pi) # Scaling the weights
    
    ###################################
    # Part 1: Deterministic problem
    # - Parents are dead, and in the next periods agents are retired
    # - There are no more income and bequest shocks => thus 'deterministic'
    # - The only risk left is mortality risk
    
    # A) Solve the final period (independent of skill levels)
    cT_vec = solveT(par) # Solve the ultimate period for optimal consumption // Over the exogenous grid par['𝒢_m_det']
    𝒢_M_det[T-1,:] =  np.append(0,𝒢_m_det) # Add a zero as first input (Savings>0)
    Cstar_det[T-1,:] = np.append(0,cT_vec) # With zero cash-on-hand, you get zero consumption

    print('Start of solving the model')
    print('0.0%')          
    # B) All the other deterministic periods
    for t in range(T-2,T-ch,-1):
        ################################################################################################################
        # 2 ) Loop over child savings (EGM)
        for ia,a in enumerate(𝒢_a_det):
            m_plus = R*a # cash-on-hand tomorrow
            C_plus = tools.interp_linear_1d_scalar(𝒢_M_det[t+1,:],Cstar_det[t+1,:],m_plus) # consumption tomorrow
            Cstar_det[t,ia+1] = o*(R*β*ψ[t+1]*(C_plus/o)**(-ρ) + (1-ψ[t+1])*κ*((a + a̲)/o)**(-γ))**(-1/ρ) # opt. consumption today / Euler 
            𝒢_M_det[t,ia+1] = a + Cstar_det[t,ia+1] # endogenous cash-on-hand grid of the deterministic periods < T 
            

    ###################################
    # Part 2: Stochastic, non-deterministic problem
    # - Parents could still be alive and agents will be in the work force in the next period
    # - Agents face income shocks and potentially bequest shocks, no mortality risk
    
    ############################################################################################################################
    #  0 ) Loop over skill levels
    for ih,h in enumerate(H): 
        Cstar_sto = np.zeros([T,gridsize_z,gridsize_m,2,gridsize_deterministic_m]) # storage for Cstar for a specific skill level h, later appended to list H_Cstar
        𝒢_MEn_sto = np.zeros([T,gridsize_z,gridsize_m,2,gridsize_deterministic_m]) # storage for cash-on-hand grid for a specific skill level h, later appended to list H_𝒢_MEn
                    
        ############################################################################################################################
        #  1 ) Loop over last period productivity level
        for iz,z in enumerate(𝒢_z):
            # Gauss-Hermite draws
            ez = np.exp(σ_z*np.sqrt(2)*x + α*z) # e^{z_t} = np.exp(σ*np.sqrt(2)*x + rho*z_{t-1} ) see slides
            ez_shock = ez.flatten() # S shock nodes
            
            ############################################################################################################################
            # 2 ) Loop backwards over time periods 
            for t in range(T-ch,-1,-1):
                # with T = 6 and ch=4 we have t=2,1,0
                A_max_h = h*w*np.sum(l[0:(t+1)])*stats.lognorm.ppf(0.99,σ_z) # the t times 99 percentil would have this amount of cash-on-hand to potentionally be saved
                𝒢_a_h = np.linspace(10e-6,A_max_h,gridsize_m-1) # constant gridsize even though M_max_a_h changes with h and z
                        
                ########################################################################################################################
                # 3 ) Loop over parents cash-on-hand
                for imp,mp in enumerate(𝒢_M_det[t+ch-1,:]): # parents are in the deterministic periods of life
                    Cparents = Cstar_det[t+ch-1,imp] # Cstar[T-1] and Cstar[T-2] is independent of grandparents wealth, because they have died
                    ap = mp - Cparents # savings
                    mp_plus = np.repeat(R*ap,S) # cash-on-hand tomorrow
                    b = mp_plus # potential bequest
                    ####################################################################################################################
                    # 4 ) Parents Dead or Alive?
                    for φ in np.array([0,1]):
                        ################################################################################################################
                        # 5 ) Loop over child savings
                        for ia,a in enumerate(𝒢_a_h):
                            m_plus = w*h*l[t+1]*ez_shock + R*a + b*np.array([np.array([0,1]),]*S).T # cash-on-hand tomorrow dim=[2,S] first row without bequest, second row with bequest
                            if φ==0: # parents haven't died yet and we might get bequest tomorrow   
                                if t==T-ch: # t=2, Age=3 // we know our parents (if they are still alive) will die in t==3
                                    C_plus = tools.interp_linear_1d(𝒢_M_det[t+1,:],Cstar_det[t+1,:],m_plus[1,]) # consumption tomorrow given cash-on-hand tomorrow 'm_plus'
                                    Cstar_sto[t,iz,ia+1,φ,imp] = (R*β*ψ[t+1] *(ω @ C_plus**(-ρ)))**(-1/ρ) # optimal consumption in stochastic period for given m,z,mp
                                else: # t=0,1, Age=1,2
                                    # parents wealth tomorrow 'mp_plus' can jump off the endogous grid 𝒢_M_det[t+ch,:]
                                    # This is why we are doing a binary search on the grid and then take a weighted average of corresponding 
                                    # optimal consumption tomorrow if parents survive
                                    imp_plus_min = tools.binary_search(0,gridsize_deterministic_m,𝒢_M_det[t+ch,:],mp_plus[0]) # between which two gridpoints does parents wealth fall?
                                    imp_plus_max = imp_plus_min + 1
                                    
                                    # caluclation of consumption tomorrow given cash-on-hand tomorrow 'm_plus'if parents survive
                                    C_plus0_min = tools.interp_linear_1d(𝒢_MEn_sto[t+1,iz,:,0,imp_plus_min],Cstar_sto[t+1,iz,:,0,imp_plus_min],m_plus[0,]) # If parent survives
                                    C_plus0_max = tools.interp_linear_1d(𝒢_MEn_sto[t+1,iz,:,0,imp_plus_max],Cstar_sto[t+1,iz,:,0,imp_plus_max],m_plus[0,]) # If parent survives
                                    weight = (mp_plus[0] - 𝒢_M_det[t+ch,imp_plus_min])/(𝒢_M_det[t+ch,imp_plus_max] - 𝒢_M_det[t+ch,imp_plus_min])
                                    C_plus0 = (1-weight)*C_plus0_min + weight*C_plus0_max
                                    
                                    # caluclation of consumption tomorrow given cash-on-hand tomorrow 'm_plus'if parents have died
                                    C_plus1 = tools.interp_linear_1d(𝒢_MEn_sto[t+1,iz,:,1,0],Cstar_sto[t+1,iz,:,1,0],m_plus[1,]) # if parent has just died
                                    
                                    # calulation of consumption today given expected consupmtion tomorrow
                                    exp_uprime =  ψ[t+ch]*ω @ C_plus0**(-ρ) + (1-ψ[t+ch])*ω @ C_plus1**(-ρ) 
                                    Cstar_sto[t,iz,ia+1,φ,imp] = (R*β*ψ[t+1]*exp_uprime)**(-1/ρ) ## optimal consumption today in stochastic period for given m,z,mp
                            else: # φ==1 // We have already received bequest because parents are dead
                                if t==T-ch: # t=2, Age=3 // we have to use the deterministic grid
                                    C_plus = tools.interp_linear_1d(𝒢_M_det[t+1,:],Cstar_det[t+1,:],m_plus[0,]) # consumption tomorrow given cash-on-hand  
                                else: # t=0,1, Age=1,2 // we have to use the endogenous grid
                                    C_plus = tools.interp_linear_1d(𝒢_MEn_sto[t+1,iz,:,1,0],Cstar_sto[t+1,iz,:,1,0],m_plus[0,]) # consumption tomorrow given cash-on-hand 
                                Cstar_sto[t,iz,ia+1,φ,imp] = (R*β*ψ[t+1]*ω @ (C_plus**(-ρ)))**(-1/ρ)
                            𝒢_MEn_sto[t,iz,ia+1,φ,imp] = a + Cstar_sto[t,iz,ia+1,φ,imp] # endogenous grid
            
                            
        H_Cstar.append(Cstar_sto)
        H_𝒢_MEn.append(𝒢_MEn_sto)
        print(str((1+ih)*100/len(H)) + '%')
    print('Model is solved')

    return Cstar_det,𝒢_M_det,H_Cstar,H_𝒢_MEn


def sim(𝒢_M_det,Cstar_det,H_𝒢_MEn,H_Cstar,par,N,b_iter): # Simulation of the model
    σ_eps = par['σ_eps'] # For conditional
    σ_z = par['σ_z'] # For unconditional
    α = par['α']
    ψ = par['ψ']
    T = par['T']
    r = par['r']
    ch = par['ch']
    R = par['R']
    l = par['l']
    H = par['H']
    w = par['w']
    q = par['q']
    gridsize_m = par['gridsize_m']
    gridsize_deterministic_m = par['gridsize_deterministic_m']
    gridsize_z = par['gridsize_z']
    P_markov = par['P_markov']
    𝒢_z = par['𝒢_z']
    
    print('Start of model simulation')
    print('0.0%') 
    
    # Setting the seed.
    np.random.seed(2021)
    
    # Allocating skill types to simulated agents / each skill should be represented according to pop shares par['q']
    hi_children = []
    for ih,h in enumerate(H): # loop over skills
        hi_children.append(np.repeat(ih,np.floor(q[ih]*N))) 
    hi_children = np.hstack(hi_children)
    mc = qe.MarkovChain(P_markov.T, state_values=np.arange(len(H))) # create a markov chain method based on P_markov to be used later
    
    # Preallocation 
    N = len(hi_children)
    MPguess = np.zeros([T+1,N]) # guess cash-on-hand distribution
    simM = np.nan + np.zeros([T+1,N]) # storage for the simulation of cash-on-hand
    simC = np.nan + np.zeros([T,N]) # storage for the simulation of consumption
    simB = np.nan + np.zeros([T,N]) # storage for the simulation of bequest
    simA = np.zeros([T+1,N]) # storage for the simulation of savings
    simZ = np.zeros([T+1,N]) # storage for the simulation of the productivity parameter z
    simY = np.zeros([T+1,N]) ## Put this within the loop!!!
    
    simMean = np.zeros([b_iter,1]) # storage for mean simM at retirement / we store it to evaluate convergence
    simStd = np.zeros([b_iter,1]) # storage for mean simM at retirement / we store it to evaluate convergence
    
    JustDead,DOA = simMortality(N,ψ) # Draw Initial mortality pattern of parents
       
    ############################################################################################################################
    #  0 ) Loop over convergences iterations
    for j in range(b_iter):
        # Filling out skill levels of children
        h_children = np.zeros(N)       
        hi_parents = hi_children # Initializing // Children will also become parents
        hi_children = mc.simulate(ts_length=2, init=hi_parents)[:,1]     
        h_children = np.take(H,hi_children) # from index to explicit h value
        
        # Initialize productivity and cash-on-hand in period t=0
        simZ[0,:] = np.random.normal(0,σ_z,N) # From the Ergodic Distribution
        simY[0,:] = l[0]*h_children*np.exp(simZ[0,:]) # E[y] = exp(μ + σ_z**2/2) // Initial Productivity from Ergodic Dist.
        simM[0,:] = w*simY[0,:] # Cash-on-hand at birth
        
        #######################################################################
        # 1) Loop over time-periods of life        
        for t in range(T):
            # Draw Productivity Shock in the next period
            if t<r-1: # Only compute shocks for periods in the labor market
                simZ[t+1,:] = np.random.normal(α*simZ[t,:],σ_eps,N) # From the conditional distribution
                simY[t+1,:] = l[t+1]*h_children*np.exp(simZ[t+1,:])
            
            ############################################################################################################################
            #  2) Loop over children-parent relation
            for i,h in enumerate(h_children):
                φi = DOA[i,t+ch-1]
                
                𝒢_MEn_sto = H_𝒢_MEn[hi_children[i]]
                Cstar_sto = H_Cstar[hi_children[i]]
    
                if t<=T-ch: # Our parents could still be alive // t=0,1,2, age=1,2,3
                    mp = MPguess[t+ch-1,i] # cash-on-hands of parents
                    
                    # Binary Search for Parents Cash-on-hand
                    ip_min = tools.binary_search(0,gridsize_m,𝒢_M_det[t+ch-1,:],mp)
                    ip_max = ip_min + 1
                    
                    # Binary Search for Child Productivity
                    iz_min = tools.binary_search(0,gridsize_z,𝒢_z,simZ[t,i])
                    iz_max = iz_min + 1
                    
                    
                    # First step: Varying Z given ip_min and ip_max
                    
                    if ip_min == gridsize_deterministic_m - 1 and iz_min == gridsize_z - 1:
                        simC[t,i] = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_min,:,0,ip_min],Cstar_sto[t,iz_min,:,φi,ip_min],simM[t,i])
                        
                    elif ip_min == gridsize_deterministic_m - 1 and iz_min < gridsize_z - 1:
                        weight_Z = (simZ[t,i] - 𝒢_z[iz_min])/(𝒢_z[iz_max] - 𝒢_z[iz_min])
                        simC_min = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_min,:,φi,ip_min],Cstar_sto[t,iz_min,:,φi,ip_min],simM[t,i])
                        simC_max = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_max,:,φi,ip_min],Cstar_sto[t,iz_max,:,φi,ip_min],simM[t,i])
                        simC[t,i] = simC_min*(1-weight_Z) + simC_max*weight_Z
                        
                        
                    elif ip_min < gridsize_deterministic_m - 1 and iz_min == gridsize_z - 1:
                        weight = (mp - 𝒢_M_det[t+ch-1,ip_min])/(𝒢_M_det[t+ch-1,ip_max] - 𝒢_M_det[t+ch-1,ip_min])
                        simC_min = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_min,:,φi,ip_min],Cstar_sto[t,iz_min,:,φi,ip_min],simM[t,i])
                        simC_max = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_min,:,φi,ip_max],Cstar_sto[t,iz_min,:,φi,ip_max],simM[t,i])
                        simC[t,i] = simC_min*(1-weight) + simC_max*weight
                        
                    else:                     
                        weight_Z = (simZ[t,i] - 𝒢_z[iz_min])/(𝒢_z[iz_max] - 𝒢_z[iz_min])
                        weight = (mp - 𝒢_M_det[t+ch-1,ip_min])/(𝒢_M_det[t+ch-1,ip_max] - 𝒢_M_det[t+ch-1,ip_min])
                        # First Stap: Varying Z for given P   
                        simC_min_Z1 = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_min,:,φi,ip_min],Cstar_sto[t,iz_min,:,φi,ip_min],simM[t,i])
                        simC_max_Z1 = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_max,:,φi,ip_min],Cstar_sto[t,iz_max,:,φi,ip_min],simM[t,i])  
                        simC_Z1 = (1-weight_Z)*simC_min_Z1 + weight_Z*simC_max_Z1 
                        simC_min_Z2 = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_min,:,φi,ip_max],Cstar_sto[t,iz_min,:,φi,ip_max],simM[t,i])
                        simC_max_Z2 = tools.interp_linear_1d_scalar(𝒢_MEn_sto[t,iz_max,:,φi,ip_max],Cstar_sto[t,iz_max,:,φi,ip_max],simM[t,i])  
                        simC_Z2 = (1-weight_Z)*simC_min_Z2 + weight_Z*simC_max_Z2 
                        
                        # Second step: Interpolating the two new points over ip_min, ip_max
                        simC[t,i] = simC_Z1*(1-weight) + simC_Z2*weight        
                    
                    Cparents = tools.interp_linear_1d_scalar(𝒢_M_det[t+ch-1,:],Cstar_det[t+ch-1,:],mp)
                    simB[t,i] = JustDead[i,t+ch]*R*(mp - Cparents)
                else: # Parents are dead for sure // t=3,4,5, age=4,5,6
                    simC[t,i] = tools.interp_linear_1d_scalar(𝒢_M_det[t,:],Cstar_det[t,:],simM[t,i])
                    simB[t,i] = np.zeros(1)
            simM[t+1,:] =  w*simY[t+1,:] + R*(simM[t,:] - simC[t,:]) + simB[t,:]
            simA[t+1,:] = simM[t,:] - simC[t,:]
        
        MPguess = simM   # update guess for cash-on-hand
        JustDead,DOA = simMortality(N,ψ) # Update mortality for children, becoming parents
        
        simMean[j] = np.mean(simM[4,:])
        simStd[j] = np.std(simM[4,:])
        print(str((1+j)*100/b_iter) + '%')

    # totA = (simA[1:T+1,:].T*(1-DOA)[:,0:T]).sum() # Total Wealth of the living
    # totB = simB.sum()*R # Total bequest    
    # BtoA = totB/totA # Bequest-to-wealth ratio       
    print('Model is simulated')
    gini,P60P80 = Moments(simM,simA,DOA,par)
    print('gini: ' +  str(gini))
    print('P60P80: ' +  str(P60P80))
    
    return simY,simZ,simM,simC,simA,simMean,simStd,hi_children,DOA # gini,P90P100 # gini,BperGDP,TsimM,TsimC,hi_children,hi_parents


def objective(θ,targets,par,N,b_iter):
    par['κ'] = θ[0] # guess
    par['a̲'] = θ[1] # guess

    target_1 = targets[0]
    target_2 = targets[1]
    
    Cstar_det,𝒢_M_det,H_Cstar,H_𝒢_MEn = solve(par)
    
    simY,simZ,simM,simC,simA,simMean,simStd,hi_children,DOA = sim(𝒢_M_det,Cstar_det,H_𝒢_MEn,H_Cstar,par,N,b_iter)
    
    JustDead,DOA = simMortality(N,par['ψ'])
    moment_1,moment_2 = Moments(simM,simA,DOA,par)
    
    W = np.array([1,1])*np.identity(2)
    L = np.array([(moment_1 - target_1)/target_1,(moment_2 - target_2)/target_2])
    Loss = L.T@W@L
    
    return Loss

def simMortality(N,ψ):
    # This function simulates mortality for all individuals in a generation
    # JustDead = 1 in the single period where the individual dies // JustDead = 0 else
    # DOA = 0 when the individual is alive // DOA = 1 if the individual is alive 
    mortal_matrix = np.random.binomial(size=[N,ψ.size],n=1,p=(1-ψ)*np.ones([N,ψ.size]))
    JustDead = np.cumsum(np.cumsum(mortal_matrix,axis=1),axis=1)
    JustDead[JustDead>1] = 0
    DOA = np.cumsum(np.cumsum(mortal_matrix,axis=1),axis=1)
    DOA[DOA>1] = 1
    return JustDead,DOA


def Moments(simM,simA,DOA,par):
    # Gini at retirement + P90-100 wealth share
    r = par['r']
    T = par['T']
    gini = f_gini(simM[r,:],100)
    
    AOD = DOA.astype('float')
    AOD[AOD==1] = np.nan
    AOD[AOD==0] = 1      # 1 if alive, NaN if Dead
    
    ActualA = simA[1:T+1,:].T*AOD[:,0:T]
    
    simA_sort = np.sort(ActualA.flatten())[~np.isnan(np.sort(ActualA.flatten()))]
    
    N_80 = int(np.round(0.8*simA_sort.size))
    N_60 = int(np.round(0.6*simA_sort.size))
    P60P80 = np.sum(simA_sort[N_60:N_80-1])/np.sum(simA_sort)
    return gini,P60P80


def plots(par,simC,simM,simA,simMean,simStd,hi_children,save_to):
    r = par['r']
    M_max = par['M_max']
    # Consumption
    Ch1 = np.mean(simC[:,hi_children==0],axis=1)
    Ch2 = np.mean(simC[:,hi_children==1],axis=1)
    Ch3 = np.mean(simC[:,hi_children==2],axis=1)
    Ch4 = np.mean(simC[:,hi_children==3],axis=1)
    Ch5 = np.mean(simC[:,hi_children==4],axis=1)
    x = np.array([1,2,3,4,5,6]) 
    
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(x,Ch1,label='H1')
    ax1.plot(x,Ch2,label='H2')
    ax1.plot(x,Ch3,label='H3')
    ax1.plot(x,Ch4,label='H4')
    ax1.plot(x,Ch5,label='H5')
    ax1.set_xlabel('Periods')
    ax1.set_ylabel('Consumption')
    ax1.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    
    plt.savefig(save_to + 'ConsumptionPath.png', dpi = 300)
    plt.show()
    plt.close()
    
    
    # Cash-On-Hand
    Mh1 = np.mean(simM[:,hi_children==0],axis=1)
    Mh2 = np.mean(simM[:,hi_children==1],axis=1)
    Mh3 = np.mean(simM[:,hi_children==2],axis=1)
    Mh4 = np.mean(simM[:,hi_children==3],axis=1)
    Mh5 = np.mean(simM[:,hi_children==4],axis=1)
    x2 = np.array([1,2,3,4,5,6,7]) 
    
    
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(x2,Mh1,label='H1')
    ax1.plot(x2,Mh2,label='H2')
    ax1.plot(x2,Mh3,label='H3')
    ax1.plot(x2,Mh4,label='H4')
    ax1.plot(x2,Mh5,label='H5')
    ax1.set_xlabel('Periods')
    ax1.set_ylabel('Wealth')
    ax1.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    
    plt.savefig(save_to + 'CashOnHandPath.png', dpi = 300)
    plt.show()
    plt.close()
    
    # Savings
    Ah1 = np.mean(simA[:,hi_children==0],axis=1)
    Ah2 = np.mean(simA[:,hi_children==1],axis=1)
    Ah3 = np.mean(simA[:,hi_children==2],axis=1)
    Ah4 = np.mean(simA[:,hi_children==3],axis=1)
    Ah5 = np.mean(simA[:,hi_children==4],axis=1)
    x2 = np.array([0,1,2,3,4,5,6]) 
    
    
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(x2,Ah1,label='H1')
    ax1.plot(x2,Ah2,label='H2')
    ax1.plot(x2,Ah3,label='H3')
    ax1.plot(x2,Ah4,label='H4')
    ax1.plot(x2,Ah5,label='H5')
    ax1.set_xlabel('Periods')
    ax1.set_ylabel('Wealth')
    ax1.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    
    plt.savefig(save_to + 'SavingsPath.png', dpi = 300)
    plt.show()
    plt.close()
    
    
    # Wealth distribution at retirement   
    plt.tick_params(axis='x',labelleft=False)
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.hist(simM[r,:],bins = 100,range=(0,M_max))
    ax1.set_xlabel('Wealth at retirement')
    ax1.set_ylabel('Frequency')
    plt.savefig(save_to + 'WealthDist.png', dpi = 300)
    plt.show()
    plt.close()
    
    
    # Convergence of mean cash-on-hand
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(simMean)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mean of cash-on-hand at retirement')
    plt.savefig(save_to + 'MeanConvergence.png', dpi = 300)
    plt.show()
    plt.close()
    
    # Convergence of standard deviation of cash-on-hand
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(simStd)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Std. of cash-on-hand at retirement')
    plt.savefig(save_to + 'StdConvergence.png', dpi = 300)
    plt.show()
    plt.close()
    

    
def Table(par,simM,N):
    ## Table for PSID comparison
    ## We do not account for deaths as they occur randomly (independent of income)
    ## Cash-on-hand at t=r, is what the retiree has to use while retired
    r = par['r']
    simM_sort = np.sort(simM[r,:])
    
    # 40-60
    N_40 = int(np.round(N*0.4))
    N_60 = int(np.round(N*0.6))
    N_80 = int(np.round(N*0.8))
    N_90 = int(np.round(N*0.9))
    N_95 = int(np.round(N*0.95))
    N_99 = int(np.round(N*0.99))
    
    P_40 = np.sum(simM_sort[0:N_40])/np.sum(simM[r,:])
    P_4060 = np.sum(simM_sort[N_40:N_60])/np.sum(simM[r,:])
    P_6080 = np.sum(simM_sort[N_60:N_80])/np.sum(simM[r,:])
    P_8090 = np.sum(simM_sort[N_80:N_90])/np.sum(simM[r,:])
    P_9095 = np.sum(simM_sort[N_90:N_95])/np.sum(simM[r,:])
    P_9599 = np.sum(simM_sort[N_95:N_99])/np.sum(simM[r,:])
    P_99100 = np.sum(simM_sort[N_99:N-1])/np.sum(simM[r,:])
    
    table = [P_40,P_4060,P_6080,P_8090,P_9095,P_9599,P_99100]
    return table