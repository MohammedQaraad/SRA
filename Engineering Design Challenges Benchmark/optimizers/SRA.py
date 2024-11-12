# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:39:47 2024

@author: Dr. Nazar
"""

import random
import numpy
import numpy as np
import math
from solution import solution
import time
from numpy import abs, zeros, log10, where, arctanh, tanh
from numpy.random import uniform, standard_cauchy
from constant import benchmark_function as BF

def SRA(objf, lb, ub, dim, PopSize, iters,function_name):
    # PSO parameters
    i = function_name
    if i == 6:
        lb=[1,1,1,1,1,30,30,30,30,30]
        ub=[5,5,5,5,5,65,65,65,65,65]
        dim = 10
    elif i == 10:
        lb=[12,12,12,12]
        up=[60,60,60,60]
        dim = 4;
    else:
        out = BF(i)  # Get object contain information about problems
        dim, g, h, lb, ub = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
 
    L=0.5
    h=6.625*10**(-34)
    Cost=numpy.full(PopSize,float("inf")) #record the fitness of all slime mold
    pos = numpy.zeros((PopSize, dim))
    Psai =  numpy.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
    
    for i in range(0,PopSize):
        for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])
                Psai[i,j]=math.sqrt(2/L)*numpy.sin(pos[i,j])*numpy.exp(2)
        Cost[i] = objf(pos[i, :])

    SmellOrder = numpy.sort(Cost)  #Eq.(2.6)
    SmellIndex=numpy.argsort(Cost)
    Worst_Cost = SmellOrder[PopSize-1];
    Best_Cost = SmellOrder[0];
    sorted_population=pos[SmellIndex,:]
    Best_X=sorted_population[0,:]
#     ########### Sort Psai
    sorted_Psai=Psai[SmellIndex,:]
    Best_Psai=sorted_Psai[0,:]
    Worst_Psai=sorted_Psai[PopSize-1,:]

    Best_Cost =  SmellOrder[0]
    Best_X = sorted_population[0,:]        # Determine the vale of Best Fitness
    Worst_Cost =  SmellOrder[PopSize-1]
        

    convergence_curve = numpy.zeros(iters)

    ############################################
    s = solution()
    print('SRA is optimizing  "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
    
        b = 1 - (l ** (1.0 / 5)) / (iters ** (1.0 / 5)) 
        
        SmellOrder = numpy.sort(Cost)  
        SmellIndex=numpy.argsort(Cost)
        sorted_Psai=Psai[SmellIndex,:]
        Best_Psai=sorted_Psai[0,:]
        Worst_Psai=sorted_Psai[PopSize-1,:]

        Seq = numpy.array(range(0,PopSize))
        R = PopSize-Seq;
        p = (R/PopSize)**2;
           
        for i in range(0, PopSize):
            h2=(p[i])
            Xnew   =  numpy.zeros(dim)
            vc = numpy.random.uniform(-b, b, dim) 
            Z=Levy(dim)
            k=1
            if random.random() < 0.03 :
                Xnew = numpy.random.uniform(0, 1, dim)  # Eq. (26)
            else :
                ids_except_current = [_ for _ in range(PopSize) if _ != i]
                id_1, id_2  = random.sample(ids_except_current, 2) 
                ###############################
                # Explotation phase
                if numpy.abs(p[i]) >= 0.5:   
                    if numpy.random.rand() < 0.5:
                        pos[i,:]  = k*random.random()+ 2 * pos[i,:] - pos[i-1,:]      # Eq.(19)
                    else:
                        
                        Xnew =  Best_X - 0.1*Z + numpy.random.rand()*((numpy.array(ub) - 
                                numpy.array(lb))* numpy.random.rand() + numpy.array(lb))    # Eq.(20)
                #Exploration phase
                else :
                    pos_1 = Best_X + random.random() * vc *(h*(Best_Psai-Worst_Psai)+(h2*(Psai[id_1,:]-
                                    2*Psai[i,:]+Psai[id_2,:])))/Psai[i,:]            # Eq.(23)

                    pos_2 = pos[i,:] + random.random() *vc * (h*(Best_Psai-Worst_Psai)+
                                 (h2*(Psai[id_1,:]+2*Psai[i,:]+Psai[id_2,:])))/Psai[i,:]      # Eq.(24)
                    Xnew = where(uniform(0, 1) < p[i], pos_1, pos_2)

            if np.random.rand() < np.random.rand():
#                 pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
                for j in range(dim):
                    if Xnew[j] > ub[j]:
                        Xnew[j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                    elif Xnew[j] < lb[j]:
                        Xnew[j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
            else:
                Xnew=numpy.clip(Xnew, lb, ub)                    
                    
                    
#             Xnew=numpy.clip(Xnew, lb, ub)
            
            Xnew_Cost=objf(Xnew)
            if Cost[i] > Xnew_Cost:
                Cost[i]=Xnew_Cost 
                pos[i,:]=Xnew
                if Cost[i]<Best_Cost:
                    Best_X=pos[i,:]
                    Best_Cost=Cost[i]
            if Cost[i] > Worst_Cost:
                Worst_Cost = Cost[i]                    
                    
            Psai[i,:]=math.sqrt(2/L)*numpy.sin(random.random()*(pos[i,:]))*numpy.exp(2)   # Eq.(22) 
             
  
#         if l % 1 == 0:
#              print(["At iteration " + str(l)+ " the best fitness is "+ str(Best_Cost)])
   
        convergence_curve[l] = Best_Cost


    
  
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "SRA"
    s.objfname = objf.__name__


    return s


def _mutation__(current_pos, new_pos,dim,crossover_ratio,lb,ub):
    pos_new = numpy.where(numpy.random.uniform(0, 1, dim) < crossover_ratio, current_pos, new_pos)
    return pos_new
def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step