# Solving, simulating and estimating OLG models with linked bequest and intergenerational skill transmission   


REQUIREMENTS
-----------------
This module requires the following:
 * Python (3.7)
 * Quantecon package for Python 
 	* If you do not use Anaconda follow this: https://quantecon.org/quantecon-py/
	
	* If you are using Anaconda:
		1. Type into cmd:
		```
		pip install quantecon
		```
		2. Type into your Python compiler prompt:
		```
		!conda install -y quantecon
		```
		
OVERVIEW OF FILES (scripts)
-----------------
 * Main.py:
	* Runs the whole code.
	* Defines the exogenous and estimated parameter values in "par".
	* From Model.py, Main.py calls solve(), sim(), and estimates parameters by optimizing objective().
 
 * Model.py:
	* solveT():       Solves the final period of the model as a starting point for backward induction (BI).
	* solve():        Solves every period by backward induction and EGM. Calls solveT() for starting point.
	* sim():          Simuates a set of families for # iterations, using output of solve(). Calls simMortality().
	* objecive():     Defines a 2D objective function for estimation of parameters. Calls moments().
	* moments():      Computes moments to be caompared to targets in objective().
	* gini():         Computes the Gini coefficient of any vector, at a given level of stratification.
	* simMortality(): Simulates the mortality pattern of a single generation. Is called by sim() and moments().
	* plots():	  Creates relevant plots for simulation results, including convergence plots.
	* Table():	  Creates a table that compares model moments to empirical moments.

 * tools.py: (By Jeppe Druedahl https://sites.google.com/view/jeppe-druedahl/home)
	* Contains helper functions that support the funcions of Model.py.

GUIDE TO RUN
-----------------
1. Make sure that requirements are met.
2. Set exogenous parameters (or if desired estimated parameters).
	* Note that T, r, and ch cannot be changed freely. Any configuration of these must be consistent with assumptions (1)-(3) in Section 2.1. 
3. Run solve() to produce policy functions.
4. Run sim() to simulate an economy given policy functions.
5. If desired, estimate the model by setting target moments and starting values, and running "solution = optimize.minimize(...)". The solution can

Results
-----------------

### Consumption path
<img src="https://github.com/TimDominikMaurer/OLGLinkedBequest/blob/main/figtabs/CashOnHandPath.png" width="400"/>
