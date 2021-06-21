# coding=utf-8 
from __future__ import division
from scipy import integrate,stats,special
#from xlrd import open_workbook
import numpy as np
#import scipy as sp
import pylab as pl
import pytwalk
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
#import emcee
#import time
#import math
#from scipy.special import gamma
import corner
import seaborn as sns

plt.style.use('seaborn-talk') # beautify the plots!



# Model: SsEIQRP



TotalNumIter =  600000
burnin       =  300000
LastNumIter  =    3000
NumEqs       =       7
NumParams    =      12





    
    

    
data=pd.read_csv('covidMexicoNew.csv')
   
mu          = 0.000046948
nu          = 0.00001589
tau         = 0.071428571  #(1/tau_Q := period of quarantined) 
alfa        = 0.196078431

   
N           = 128932753

save_results_to = 'SEAIRD-t-walk/'
  
       


# # HUC
Suspect = data["Suspects"]
Sick = data["Cases"]
Deaths  = data["Deaths"] 

# datahub
# Sick = data["New cases (datahub)"]
# Deaths  = data["Deaths (datahub)"] 

## OWD
#Sick = data["New cases (OWD)"]
#Deaths  = data["Deaths (OWD)"] + data["Recovered (noData)"]

# # DGE
# Sick = data["New cases (DGE)"]
# Deaths  = data["Deaths (DGE)"] 
ttime  = np.linspace(0.0,float(len(Sick))-1,len(Sick))
t_pred = np.linspace(0.0,float(len(Sick))-1,len(Sick))
times_pred = np.linspace(0.0,229,230)

n_days=len(Sick)
n_pred=len(times_pred)

fig0= plt.figure()
plt.stem(ttime, Suspect, linefmt='yellow', markerfmt=" "  )
plt.stem(ttime, Sick, linefmt='mediumblue', markerfmt=" " )
plt.stem(ttime, Deaths, linefmt='orangered',markerfmt=" ")
plt.savefig(save_results_to + 'NewCases, Deaths, Suspects.eps')





# Initial conditions

# s0 = p[6]
# E0 = p[7]
# I0 = p[8]
# Q0 = p[9]
R0 = 0.
D0 = 0.
# S0 = N-(s0 + E0 + I0 + Q0 + R0 + P0) 
    


def modelo(t,x,p):
    
  """
  model:SEAIRD
  
  Parameters:
  p[0]: beta_s
  p[1]: beta_a
  p[2]: rho proportion of asymptomatics and symp
  p[3]: gamma recovery rate
  p[4]: sigma death rate by disease 
  p[5]: fraction factor due to contact tracing 
  
  State variables: SsEIQRP
  
  x[0]: Susceptibles
  x[1]: suspects
  x[2]: Exposeds
  x[3]: Asymp infected
  x[4]: Symp infected
  x[5]: Recovereds
  x[6]: Deaths
  """
  
  fx = np.zeros(NumEqs)
  
  fx[0] =  -( (1-p[5])*p[0]*x[4] + p[5]*p[0]*x[4] + p[1]*x[3] )*x[0]/N + tau*x[1]
  fx[1] =  p[5]*p[0]*x[4]*x[0]/N - tau*x[1]
  fx[2] =   ( (1-p[5])*p[0]*x[4] + p[1]*x[3] )*x[0]/N  -alfa*x[2]
  fx[3] =  p[2]*alfa*x[2]- p[3]*x[3]
  fx[4] =  (1-p[2])*alfa*x[2] -  (p[3] + p[4])* x[4]
  fx[5] =  p[3]*(x[3]+x[4])
  fx[6] =  p[4]*x[4]
  
  return fx




def solve(p):
    x0 = np.array([N-(p[6] + p[7] + p[8] + R0 + D0),0.,p[6],p[7],p[8],R0,D0]) # SEAIRD
   
    result_s = np.zeros(len(ttime))   
    result_I = np.zeros(len(ttime))
    result_D = np.zeros(len(ttime))
#    soln = integrate.odeint(modelo,x0,ttime,args=(p,))            
    soln = integrate.solve_ivp(lambda t,x: modelo(t, x, p),
                                [0,n_days],x0,method='LSODA', 
                                vectorized=True,t_eval=ttime)
    result_s=soln.y[1][:]
    result_I=soln.y[4][:]
#    result_D=soln.y[6][:]
    for k in range(len(ttime)):
        if k==0:          
           result_D[k]= soln.y[6][k]
        else:   
           result_D[k]=soln.y[6][k] -soln.y[6][k-1]
    
    return result_s,result_I,result_D



def solve_pred(p):
    x0 = np.array([N-(p[6] + p[7] + p[8] + R0 + D0),0.,p[6],p[7],p[8],R0,D0]) # SEAIRD
   
    result_s = np.zeros(len(times_pred))   
    result_I = np.zeros(len(times_pred))
    result_D = np.zeros(len(times_pred))
#    soln = integrate.odeint(modelo,x0,ttime,args=(p,))            
    soln = integrate.solve_ivp(lambda t,x: modelo(t, x, p),
                                [0,n_pred],x0,method='LSODA', 
                                vectorized=True,t_eval=times_pred)
    result_s=soln.y[1][:]
    result_I=soln.y[4][:]
#    result_D=soln.y[5][:]
    for k in range(len(times_pred)):
        if k==0:          
           result_D[k]= soln.y[6][k]
        else:   
           result_D[k]=soln.y[6][k] -soln.y[6][k-1]
    
    return result_s,result_I,result_D
   
      
    
    
    
def energy(p):
    if support(p):
        my_soln_s,my_soln_I,my_soln_D = solve(p)        

#        log_likelihood1 = -np.sum(my_soln_s-Suspect*np.log(my_soln_s))
#        log_likelihood1 = np.sum(Suspect*np.log(my_soln_s)
#                            -(p[10] + Suspect)*np.log(p[10] + my_soln_s) ) \
#             -len(Suspect)*(-p[10]*np.log(p[10])+ np.log(gamma(p[10]) )  ) 
#        log_likelihood2 = np.sum(np.log( ss.nbinom.pmf(Suspect,p[10],
#                                         p[10]/(p[10] + my_soln_Q )) ) )
        log_likelihood1 = np.sum(Suspect*np.log(my_soln_s) \
                            -(p[9] + Suspect)*np.log(p[9] + my_soln_s) ) \
             -len(Suspect)*(-p[9]*np.log(p[9])+ np.log(special.gamma(p[9]) )  )
        log_likelihood2 = np.sum(Sick*np.log(my_soln_I) \
                            -(p[10] + Sick)*np.log(p[10] + my_soln_I) ) \
             -len(Sick)*(-p[10]*np.log(p[10])+ np.log(special.gamma(p[10]) )  )
#        log_likelihood2 = -np.sum(my_soln_D-Deaths*np.log(my_soln_D)) 
        log_likelihood3 = np.sum(Deaths*np.log(my_soln_D)
                            -(p[11] + Deaths)*np.log(p[11] + my_soln_D) ) \
             -len(Deaths)*(-p[11]*np.log(p[11])+ np.log(special.gamma(p[11]) )  ) 
        #log_likelihood = -np.sum(np.linalg.norm(my_soln*N-flu_data))**2/10.0**2
#        print(log_likelihood)
       # gamma distribution parameters for p[0] = beta
        k0 = 1.0
        theta0 = 1.0
        # gamma distribution parameters for p[1] = qu
        k1 = 1.0
        theta1 = 1.0
        # gamma distribution parameters for p[2] = delta
        k2 = 1.0
        theta2 = 1.0
        # gamma distribution parameters for p[3] = alpha
        k3 = 1.0
        theta3 = 1.0
        # gamma distribution parameters for p[4] = gamma
        k4 = 1.0
        theta4 = 1.0
        # gamma distribution parameters for p[5] = sigma
        k5 = 1.0
        theta5 = 1.0
        
        k6 = 1.0
        theta6 = 10.0
        # gamma distribution parameters for p[4] = gamma
        k7 = 1.0
        theta7 = 10.0
        # gamma distribution parameters for p[5] = sigma
        k8 = 7.5
        theta8 = 1.
        
        k9 = 7.5
        theta9 = 1.0
        
        k10 = 7.5
        theta10 = 1.0

        k11 = 7.5
        theta11 = 1.0

    
        a0 = ss.lognorm.pdf(p[0], 1.)
        a1 = ss.lognorm.pdf(p[1], 1.)
#        a0 = (k0-1)*np.log(p[0])- (p[0]/theta0)
#        a1 = (k1-1)*np.log(p[1])- (p[1]/theta1)
        a2 = (k2-1)*np.log(p[2])- (p[2]/theta2) 
        a3 = (k3-1)*np.log(p[3])- (p[3]/theta3)
        a4 = (k4-1)*np.log(p[4])- (p[4]/theta4)
#        a5 = (k5-1)*np.log(p[5])- (p[5]/theta5)
        a5 = ss.uniform.pdf(p[5], 0., 0.3)
        a6 = ss.uniform.pdf(p[6], 0., 4e4)
        a7 = ss.uniform.pdf(p[7], 0., 8e3)
        a8 = ss.uniform.pdf(p[8], 0., 1e2)
        a9 = (k9-1)*np.log(p[9])- (p[9]/theta9)
        a10 =(k10-1)*np.log(p[10])- (p[10]/theta10)
        a11 = (k11-1)*np.log(p[11])- (p[11]/theta11)
#        a8 = ss.uniform.pdf(p[8], 0., 0.3)
#        a5 = (k5-1)*np.log(p[5])- (p[5]/theta5)
#        a6 = (k6-1)*np.log(p[6])- (p[6]/theta6)
#        a7 = (k7-1)*np.log(p[7])- (p[7]/theta7)

#        a11= (k11-1)*np.log(p[11])- (p[11]/theta11)
#        a12= (k12-1)*np.log(p[12])- (p[12]/theta10)
 
        log_prior = a0 + a1 + a2 + a3 + a4 + a5 + a6 + \
                    a7 + a8 + a9 + a10 + a11
        return -log_likelihood1 -log_likelihood2 -log_likelihood3 -log_prior
    return -np.infty








def support(p):
    rt = True
    rt &= (0.0 < p[0] < 0.3)  #beta_s
    rt &= (0.0 < p[1] < 0.7)  #beta_a
    rt &= (0.7 < p[2] < 1.0)  #rho
    rt &= (0.0 < p[3] < 0.2)  #gama
    rt &= (0.0 < p[4] < 0.4)  #sigma
    rt &= (0.0 < p[5] < 0.4)  #qu
    rt &= (0.0 < p[6] < 4e4)  #E0
    rt &= (0.0 < p[7] < 8e3)  #A0
    rt &= (0.0 < p[8] < 1e2)  #I0
    rt &= (5e-7 < p[9] < 0.3)  #omega
    rt &= (5e-7 < p[10] < 0.3)  #omega
    rt &= (5e-7 < p[11] < 0.3)  #omega


    return rt

def init():
    p = np.zeros(NumParams)
    p[0] = np.random.uniform(low=0.0,high=0.3) #beta_s
    p[1] = np.random.uniform(low=0.0,high=0.7) #beta_a
    p[2] = np.random.uniform(low=0.7,high=1.0) #rho
    p[3] = np.random.uniform(low=0.0,high=0.2) #gama
    p[4] = np.random.uniform(low=0.0,high=0.4) #sigma
    p[5] = np.random.uniform(low=0.0,high=0.4) #qu
    p[6] = np.random.uniform(low=0.0,high=4e4) #E0
    p[7] = np.random.uniform(low=0.0,high=8e3) #A0
    p[8] = np.random.uniform(low=0.0,high=1e2) #E0
    p[9] = np.random.uniform(low=5e-7,high=0.3) #omega
    p[10] = np.random.uniform(low=5e-7,high=0.3) #omega
    p[11] = np.random.uniform(low=5e-7,high=0.3) #omega


    return p


def euclidean(v1, v2):
    return sum((q1-q2)**2 for q1, q2 in zip(v1, v2))**.5

if __name__=="__main__": 
#    nn = len(flu_ttime)
#    print(nn)
#    input("Press Enter to continue...") 
#    burnin = 5000
     sir = pytwalk.pytwalk(n=NumParams,U=energy,Supp=support)
     sir.Run(T=TotalNumIter,x0=init(),xp0=init())

ppc_samples_s = np.zeros((LastNumIter,len(times_pred)))
ppc_samples_I = np.zeros((LastNumIter,len(times_pred)))
ppc_samples_D = np.zeros((LastNumIter,len(times_pred)))
    
    
   
    
fig0= plt.figure()
ax0 = plt.subplot(111)
sir.Ana(start=burnin)
plt.savefig(save_results_to + 'trace_plot.eps')

plt.figure()
ax2 = plt.subplot(111)
qq = sir.Output[sir.Output[:,-1].argsort()] # MAP
my_soln_s,my_soln_I, my_soln_D = solve_pred(qq[0,:]) # solve for MAP
ax2.plot(times_pred,my_soln_s,'b')
ax2.plot(times_pred,my_soln_I,'g')
ax2.plot(times_pred,my_soln_D,'c')
plt.savefig(save_results_to + 'MAP.eps')

for k in np.arange(LastNumIter): # last 1000 samples
    ppc_samples_s[k],ppc_samples_I[k],ppc_samples_D[k]= \
    solve_pred(sir.Output[-k,:])
#        sample_s, sample_Q,sample_P = solve(sir.Output[-k,:]) 
    ax2.plot(times_pred,ppc_samples_s[k],"#888888", alpha=.25) 
    ax2.plot(times_pred,ppc_samples_I[k],"#888888", alpha=.25) 
    ax2.plot(times_pred,ppc_samples_D[k],"#888888", alpha=.25)

ax2.plot(ttime,Suspect,'r.')
ax2.plot(ttime,Sick,'r.')
ax2.plot(ttime,Deaths,'r.')
plt.savefig(save_results_to + 'data_vs_samples.eps')
samples = sir.Output[burnin:,:-1]
#samples[:,1] *= N
#samples[:,2] *= N
map = qq[0,:-1]
#map[1] *= N
#map[2] *= N    



median_ppc_s = np.percentile(ppc_samples_s,q=50.0,axis=0)
median_ppc_I = np.percentile(ppc_samples_I,q=50.0,axis=0)
median_ppc_D = np.percentile(ppc_samples_D,q=50.0,axis=0)

#median_ppc_I  = np.median(ppc_samples_I,axis=0)
#median_ppc_D  = np.median(ppc_samples_D,axis=0)

CriL_ppc_s = np.percentile(ppc_samples_s,q=2.5,axis=0)
CriU_ppc_s = np.percentile(ppc_samples_s,q=97.5,axis=0)

CriL_ppc_I = np.percentile(ppc_samples_I,q=2.5,axis=0)
CriU_ppc_I = np.percentile(ppc_samples_I,q=97.5,axis=0)

CriL_ppc_D = np.percentile(ppc_samples_D,q=2.5,axis=0)
CriU_ppc_D = np.percentile(ppc_samples_D,q=97.5,axis=0)


median_ppc_beta_s = np.median(samples[:,0],axis=0)
median_ppc_beta_a = np.median(samples[:,1],axis=0)
median_ppc_rho    = np.median(samples[:,2],axis=0)
median_ppc_gamma  = np.median(samples[:,3],axis=0)
median_ppc_sigma  = np.median(samples[:,4],axis=0)
median_ppc_qu     = np.median(samples[:,5],axis=0)
median_ppc_E0     = np.median(samples[:,6],axis=0)
median_ppc_A0     = np.median(samples[:,7],axis=0)
median_ppc_I0     = np.median(samples[:,8],axis=0)
median_ppc_omega1  = np.median(samples[:,9],axis=0)
median_ppc_omega2  = np.median(samples[:,10],axis=0)
median_ppc_omega3  = np.median(samples[:,11],axis=0)

   


CriL_ppc_beta_s   = np.percentile(samples[:,0],q=2.5,axis=0)
CriU_ppc_beta_s   = np.percentile(samples[:,0],q=97.5,axis=0)

CriL_ppc_beta_a   = np.percentile(samples[:,1],q=2.5,axis=0)
CriU_ppc_beta_a   = np.percentile(samples[:,1],q=97.5,axis=0)

CriL_ppc_rho    = np.percentile(samples[:,2],q=2.5,axis=0)
CriU_ppc_rho    = np.percentile(samples[:,2],q=97.5,axis=0)


CriL_ppc_gamma  = np.percentile(samples[:,3],q=2.5,axis=0)
CriU_ppc_gamma  = np.percentile(samples[:,3],q=97.5,axis=0)

CriL_ppc_sigma  = np.percentile(samples[:,4],q=2.5,axis=0)
CriU_ppc_sigma  = np.percentile(samples[:,4],q=97.5,axis=0)

CriL_ppc_qu     = np.percentile(samples[:,5],q=2.5,axis=0)
CriU_ppc_qu     = np.percentile(samples[:,5],q=97.5,axis=0)

CriL_ppc_E0     = np.percentile(samples[:,6],q=2.5,axis=0)
CriU_ppc_E0     = np.percentile(samples[:,6],q=97.5,axis=0)

CriL_ppc_A0     = np.percentile(samples[:,7],q=2.5,axis=0)
CriU_ppc_A0     = np.percentile(samples[:,7],q=97.5,axis=0)

CriL_ppc_I0     = np.percentile(samples[:,8],q=2.5,axis=0)
CriU_ppc_I0     = np.percentile(samples[:,8],q=97.5,axis=0)

CriL_ppc_omega1  = np.percentile(samples[:,9],q=2.5,axis=0)
CriU_ppc_omega1  = np.percentile(samples[:,9],q=97.5,axis=0)


CriL_ppc_omega2  = np.percentile(samples[:,10],q=2.5,axis=0)
CriU_ppc_omega2  = np.percentile(samples[:,10],q=97.5,axis=0)

CriL_ppc_omega3  = np.percentile(samples[:,11],q=2.5,axis=0)
CriU_ppc_omega3  = np.percentile(samples[:,11],q=97.5,axis=0)

 
print(median_ppc_beta_s)
print(median_ppc_beta_a)
print(median_ppc_rho)
print(median_ppc_gamma)
print(median_ppc_sigma)
print(median_ppc_qu)
print(median_ppc_E0)
print(median_ppc_A0)
print(median_ppc_I0)
print(median_ppc_omega1)
print(median_ppc_omega2)
print(median_ppc_omega3)
 


plt.figure()
ax2 = plt.subplot(111)
ax2.stem(ttime, Suspect, linefmt='tomato', markerfmt=" ",basefmt=" ",label="Suspects"  )
#    ax2.plot(ttime,Sick,linestyle='dashed', marker='o', color='mediumblue',label="Confirmed Cases")
ax2.plot(times_pred,my_soln_s,color='mediumvioletred', lw=1.5)
ax2.plot(times_pred,median_ppc_s,color='mediumblue', lw=1.5)
ax2.fill_between(times_pred, CriL_ppc_s, CriU_ppc_s, color='blue', alpha=0.3)
ax2.set_xlabel('Time (days)')  # Add an x-label to the axes.
ax2.legend()  # Add a legend.
plt.savefig(save_results_to + 'BandsPrediction_s.pdf')


plt.figure()
ax2 = plt.subplot(111)
ax2.stem(ttime, Sick, linefmt='tomato', markerfmt=" ",basefmt=" ",label="Confirmed Cases"  )
#    ax2.plot(ttime,Sick,linestyle='dashed', marker='o', color='mediumblue',label="Confirmed Cases")
ax2.plot(times_pred,my_soln_I,color='mediumvioletred', lw=1.5)
ax2.plot(times_pred,median_ppc_I,color='mediumblue', lw=1.5)
ax2.fill_between(times_pred, CriL_ppc_I, CriU_ppc_I, color='blue', alpha=0.3)
ax2.set_xlabel('Time (days)')  # Add an x-label to the axes.
ax2.legend()  # Add a legend.
plt.savefig(save_results_to + 'BandsPrediction_I.pdf')


plt.figure()
ax2 = plt.subplot(111)
ax2.stem(ttime, Deaths, linefmt='tomato', markerfmt=" ",basefmt=" ",label="Deaths"  )
#    ax2.plot(ttime,Deaths,linestyle='dashed', marker='o', color='orangered',label="Deaths")
ax2.plot(times_pred,my_soln_D,color='mediumvioletred', lw=1.5)
ax2.plot(times_pred,median_ppc_D,color='mediumblue', lw=1.5)
ax2.fill_between(times_pred, CriL_ppc_D, CriU_ppc_D, color='blue', alpha=0.3)
ax2.set_xlabel('Time (days)')  # Add an x-label to the axes.
ax2.legend() 
plt.savefig(save_results_to + 'BandsPrediction_P.pdf')




  

   
plt.figure()
#alpha, beta= 1.0, 1.0
data=ss.uniform.rvs(0.0,1.6,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,0],density=True)
#plt.xlim(0.0,1.0)
#plt.ylim(0.0,10.0)
plt.savefig(save_results_to + 'beta_s_prior_vs_posterior.eps')

plt.figure()
#alpha, beta= 1.0, 1.0
data=ss.uniform.rvs(0.0,1.3,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,1],density=True)
#plt.xlim(0.0,1.0)
#plt.ylim(0.0,10.0)
plt.savefig(save_results_to + 'beta_a_prior_vs_posterior.eps')

plt.figure()
alpha, beta= 1.0, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,2],density=True)
#plt.xlim(0.0,1.0)
#plt.ylim(0.0,10.0)
plt.savefig(save_results_to + 'rho_prior_vs_posterior.eps')



plt.figure()
alpha, beta= 1.0, 10.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,3],density=True)
#pl.xlim(0.0,1.0)
#pl.ylim(0.0,10.0)
pl.savefig(save_results_to + 'gamma_prior_vs_posterior.eps')


plt.figure()
alpha, beta= 1.0, 10.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,4],density=True)
plt.savefig(save_results_to + 'sigma_prior_vs_posterior.eps')


plt.figure()
alpha, beta= 1.0, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,5],density=True)
plt.savefig(save_results_to + 'qu_prior_vs_posterior.eps')

 
plt.figure()
data=ss.uniform.rvs(0,4e4,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,6],density=True)
plt.savefig(save_results_to + 'E0_prior_vs_posterior.eps')


plt.figure()
data=ss.uniform.rvs(0,8e3,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,7],density=True)
plt.savefig(save_results_to + 'A0_prior_vs_posterior.eps')

plt.figure()
data=ss.uniform.rvs(0,1e2,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,8],density=True)
plt.savefig(save_results_to + 'I0_prior_vs_posterior.eps')


plt.figure()
alpha, beta= 7.5, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,9],density=True)
plt.savefig(save_results_to + 'omega_prior_vs_posterior.eps')



    

# Define the borders
x     = samples[:,0]
y     = samples[:,1]
z     = samples[:,2]
w     = samples[:,3]
v     = samples[:,4]
vq    = samples[:,5]
e0    = samples[:,6]
a0    = samples[:,7]
i0    = samples[:,8]
ome1  = samples[:,9]
ome2  = samples[:,10]
ome3  = samples[:,11]

  
sampleT = pd.DataFrame(samples, columns=["beta_s", "beta_a","rho","gamma", "sigma","q", 
                                         "E0","A0", "I0","omega_1","omega_2","omega_3"])


plt.figure()
sns.kdeplot(data=x)
#    plt.axvline(x=CriL_ppc_beta_s,color='r',linestyle="--")
#    plt.axvline(x=CriU_ppc_beta_s,color='r',linestyle="--")
x_values = [CriL_ppc_beta_s, CriU_ppc_beta_s]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)   
plt.savefig(save_results_to + 'BayesianInterval-beta_s.eps')


plt.figure()
sns.kdeplot(data=y)
x_values = [CriL_ppc_beta_a, CriU_ppc_beta_a]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)  
plt.savefig(save_results_to + 'BayesianInterval-beta_a.eps')


plt.figure()
sns.kdeplot(data=z)
x_values = [CriL_ppc_rho, CriU_ppc_rho]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-rho.eps')


plt.figure()
sns.kdeplot(data=w)
x_values = [CriL_ppc_gamma, CriU_ppc_gamma]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-gamma.eps')


plt.figure()
sns.kdeplot(data=v)
x_values = [CriL_ppc_sigma, CriU_ppc_sigma]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-sigma.eps')



plt.figure()
sns.kdeplot(data=vq)
x_values = [CriL_ppc_qu, CriU_ppc_qu]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-sigma.eps')


plt.figure()
sns.kdeplot(data=e0)
x_values = [CriL_ppc_E0, CriU_ppc_E0]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-E0.eps')


plt.figure()
sns.kdeplot(data=a0)
x_values = [CriL_ppc_A0, CriU_ppc_A0]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-A0.eps')



plt.figure()
sns.kdeplot(data=i0)
x_values = [CriL_ppc_I0, CriU_ppc_I0]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-I0.eps')


plt.figure()
sns.kdeplot(data=ome1)
x_values = [CriL_ppc_omega1, CriU_ppc_omega1]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-omega1.eps')


plt.figure()
sns.kdeplot(data=ome2)
x_values = [CriL_ppc_omega2, CriU_ppc_omega2]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-omega2.eps')



plt.figure()
sns.kdeplot(data=ome3)
x_values = [CriL_ppc_omega3, CriU_ppc_omega3]
y_values = [0, 0]
plt.plot(x_values, y_values,color='k', lw=6)    
plt.savefig(save_results_to +  'BayesianInterval-omega3.eps')





    
print('Norm Square of Suspects data =')
print(euclidean(median_ppc_s, Suspect)) 
print('Norm Square of Sick data =')
print(euclidean(median_ppc_I, Sick))
print('Norm Square of Deaths data =')
print(euclidean(median_ppc_D, Deaths))

varnames=[r"$\beta_{s}$", r"$\beta_{a}$" , r"$\rho$"  , r"$\gamma$", r"$\sigma$" , r"$q$",
          r"$E_{0}$", r"$A_{0}$",r"$I_{0}$",r"$\omega_{1}$",r"$\omega_{2}$",r"$\omega_{3}$"]



range = np.array([(0.95*x,1.05*x) for x in map])
#corner.corner(samples,show_titles=True,labels=varnames,
#                  quantiles=[0.025, 0.5, 0.975],
#                  truths=map,range=range)
corner.corner(sampleT,show_titles=True,labels=varnames,truths=map,range=range,
                plot_datapoints=False,quantiles=[0.025, 0.5, 0.975],
                title_fmt='.4f')
plt.savefig(save_results_to + 'corner.pdf')




