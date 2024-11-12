# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 02:07:37 2021

@author: Mohanmmed
"""

from numpy.random import seed, permutation
from numpy import dot, ones, array, ceil
from utils import *
import numpy
import math
from constant import benchmark_function as BF

# define the function blocks
from numpy import zeros, array, log, abs, exp, sqrt, pi, round, sin, cos, arccos, remainder, arcsin, arctan, imag, log10
from scipy.optimize import fminbound
import constant



def F5(x):
    ## Weight Minimization of a Speed Reducer
    out = constant.benchmark_function(15)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    hx = 0
    gx = zeros(g)
    fx = 0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2) \
         + 7.477 * (x[5] ** 3 + x[6] ** 3) + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)
    PCONST=10000; 
    gx[0] = -x[0] * x[1] ** 2 * x[2] + 27
    gx[1] = -x[0] * x[1] ** 2 * x[2] ** 2 + 397.5
    gx[2] = -x[1] * x[5] ** 4 * x[2] * x[3] ** (-3) + 1.93
    gx[3] = -x[1] * x[6] ** 4 * x[2] / x[4] ** 3 + 1.93
    gx[4] = 10 * x[5] ** (-3) * sqrt(16.91 * 10 ** 6 + (745 * x[3] / (x[1] * x[2])) ** 2) - 1100
    gx[5] = 10 * x[6] ** (-3) * sqrt(157.5 * 10 ** 6 + (745 * x[4] / (x[1] * x[2])) ** 2) - 850
    gx[6] = x[1] * x[2] - 40
    gx[7] = -x[0] / x[1] + 5
    gx[8] = x[0] / x[1] - 12
    gx[9] = 1.5 * x[5] - x[3] + 1.9
    gx[10] = 1.1 * x[6] - x[4] + 1.9
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2 + gx[3]** 2 + gx[4]** 2 + gx[5]** 2 + gx[6]** 2 + gx[7]** 2 + gx[8]** 2+ gx[9]** 2 + gx[10]** 2  )
    return PHI


# def F1(x):
#     ## Tension/compression  spring  design (case 1)
#     out = constant.benchmark_function(17)
#     D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
#     hx = 0
#     fx = x[0] ** 2 * x[1] * (x[2] + 2)
#     gx = zeros(g)
#     gx[0] = 1 - (x[1] ** 3 * x[2]) / (71785 * x[0] ** 4)
#     gx[1] = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4)) + 1 / (5108 * x[0] ** 2) - 1
#     gx[2] = 1 - 140.45 * x[0] / (x[1] ** 2 * x[2])
#     gx[3] = (x[0] + x[1]) / 1.5 - 1
#     sum([sum(gv.*(gv>0)), sum(abs(hv).*(abs(hv)>input.delta))])./(g + h);
#     return fx, gx, hx


def F1(x):
    ## Tension/compression  spring  design (case 1)
    out = constant.benchmark_function(17)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    #print(" gn , hn " , g , h)
    hx = 0
    PCONST = 100
    fx = x[0] ** 2 * x[1] * (x[2] + 2)
    gx = zeros(g)
    gx[0] = 1 - (x[1] ** 3 * x[2]) / (71785 * x[0] ** 4)
    gx[1] = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4)) + 1 / (5108 * x[0] ** 2) - 1
    gx[2] = 1 - 140.45 * x[0] / (x[1] ** 2 * x[2])
    gx[3] = (x[0] + x[1]) / 1.5 - 1
    gx[gx<0] = 0
    PHI = fx + PCONST*(gx[0]**2+ gx[1]**2+gx[2]**2+gx[3]**2)
    return PHI


def F2(x):
    ## Update
    out = constant.benchmark_function(18)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x[0] = 0.0625 * round(x[0])
    x[1] = 0.0625 * round(x[1])
    ## Pressure vessel design
    PCONST=10000
    hx = 0
    gx = zeros(g)
    fx = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2] ** 2 + 3.1661 * x[0] ** 2 * x[3] + 19.84 * x[0] ** 2 * x[2]
    gx[0] = -x[0] + 0.0193 * x[2]
    gx[1] = -x[1] + 0.00954 * x[2]
    gx[2] = -pi * x[2] ** 2 * x[3] - 4 / 3 * pi * x[2] ** 3 + 1296000
    gx[3] = x[3] - 240
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2 + gx[3]** 2 )
    return PHI


def F4(x):
    out = constant.benchmark_function(19)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    ## Welded beam design
    fx = 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])
    hx = 0
    P = 6000
    L = 14
    delta_max = 0.25
    E = 30 * 1e6
    G = 12 * 1e6
    T_max = 13600
    sigma_max = 30000
    PCONST = 100000;
    Pc = 4.013 * E * sqrt(x[2] ** 2 * x[3] ** 6 / 30) / L ** 2 * (1 - x[2] / (2 * L) * sqrt(E / (4 * G)))
    sigma = 6 * P * L / (x[3] * x[2] ** 2)
    delta = 6 * P * L ** 3 / (E * x[2] ** 2 * x[3])
    J = 2 * (sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4))
    R = sqrt(x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
    M = P * (L + x[1] / 2)
    ttt = M * R / J
    tt = P / (sqrt(2) * x[0] * x[1])
    t = sqrt(tt ** 2 + 2 * tt * ttt * x[1] / (2 * R) + ttt ** 2)
    ## constraints
    gx = zeros(g)
    gx[0] = t - T_max
    gx[1] = sigma - sigma_max
    gx[2] = x[0] - x[3]
    gx[3] = delta - delta_max
    gx[4] = P - Pc
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2 + gx[3]** 2 + gx[4]** 2 )
    return PHI



def F3(x):
    out = constant.benchmark_function(20)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    ## Three-bar truss design problem
    fx = (2 * sqrt(2) * x[0] + x[1]) * 100
    gx = zeros(g)
    PCONST=10000
    gx[0] = (sqrt(2) * x[0] + x[1]) / (sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * 2 - 2
    gx[1] = x[1] / (sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * 2 - 2
    gx[2] = 1 / (sqrt(2) * x[1] + x[0]) * 2 - 2
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2)
    return PHI

def F7(x):
    ## Multiple disk clutch brake design problem
    out = constant.benchmark_function(21)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    ## parameters
    PCONST=100000
    Mf = 3
    Ms = 40
    Iz = 55
    n = 250
    Tmax = 15
    s = 1.5
    delta = 0.5
    Vsrmax = 10
    rho = 0.0000078
    pmax = 1
    mu = 0.6
    Lmax = 30
    delR = 20
    Rsr = 2 / 3 * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 * x[0] ** 2)
    Vsr = pi * Rsr * n / 30
    A = pi * (x[1] ** 2 - x[0] ** 2)
    Prz = x[3] / A
    w = pi * n / 30
    Mh = 2 / 3 * mu * x[3] * x[4] * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 - x[0] ** 2)
    T = Iz * w / (Mh + Mf)
    hx = 0
    gx = zeros(g)
    fx = pi * (x[1] ** 2 - x[0] ** 2) * x[2] * (x[4] + 1) * rho
    gx[0] = -x[1] + x[0] + delR
    gx[1] = (x[4] + 1) * (x[2] + delta) - Lmax
    gx[2] = Prz - pmax
    gx[3] = Prz * Vsr - pmax * Vsrmax
    gx[4] = Vsr - Vsrmax
    gx[5] = T - Tmax
    gx[6] = s * Ms - Mh
    gx[7] = -T
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2+ gx[3]** 2+ gx[4]** 2+ gx[5]** 2+ gx[6]** 2+ gx[7]** 2)
    return PHI




def OBJ11(x, n):
    a = x[0]
    b = x[1]
    c = x[2]
    e = x[3]
    f = x[4]
    l = x[5]
    Zmax = 99.9999
    P = 100
    if n == 1:
        def fhd(z):
            return P * b * sin(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) +
                               arccos((b ** 2 + (l - z) ** 2 + e ** 2 - a ** 2) / (2 * b * sqrt((l - z) ** 2 + e ** 2)))) / \
                   (2 * c * cos(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) + arctan(e / (l - z))))

        fhd_func = fhd
    else:
        def fhd(z):
            return -(P * b * sin(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) +
                                 arccos((b ** 2 + (l - z) ** 2 + e ** 2 - a ** 2) / (2 * b * sqrt((l - z) ** 2 + e ** 2)))) /
                     (2 * c * cos(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) + arctan(e / (l - z)))))

        fhd_func = fhd
    return fminbound(fhd_func, 0, Zmax)




def F8(x):
    ## Hydro-static thrust bearing design problem
    out = constant.benchmark_function(25)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    #PCONST=10000
    PCONST = 1000
    R = x[0]
    Ro = x[1]
    mu = x[2]
    Q = x[3]
    gamma = 0.0307
    C = 0.5
    n = -3.55
    C1 = 10.04
    Ws = 101000
    Pmax = 1000
    delTmax = 50
    hmin = 0.001
    gg = 386.4
    N = 750
    P = (log10(log10(8.122 * 1e6 * mu + 0.8)) - C1) / n
    delT = 2 * (10 ** P - 560)
    Ef = 9336 * Q * gamma * C * delT
    h = (2 * pi * N / 60) ** 2 * 2 * pi * mu / Ef * (R ** 4 / 4 - Ro ** 4 / 4) - 1e-5
    Po = (6 * mu * Q / (pi * h ** 3)) * log(R / Ro)
    W = pi * Po / 2 * (R ** 2 - Ro ** 2) / (log(R / Ro) - 1e-5)
    ##  objective function
    fx = (Q * Po / 0.7 + Ef) / 12
    ##  constraints
    gx = zeros(g)
    hx = 0
    gx[0] = Ws - W
    gx[1] = Po - Pmax
    gx[2] = delT - delTmax
    gx[3] = hmin - h
    gx[4] = Ro - R
    gx[5] = gamma / (gg * Po) * (Q / (2 * pi * R * h)) - 0.001
    gx[6] = W / (pi * (R ** 2 - Ro ** 2) + 1e-5) - 5000
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2 + gx[3]** 2+ gx[4]** 2+ gx[5]** 2+ gx[6]** 2)
    return PHI


def F9(x):
    #  Process design Problem
    #  Himmelblau's Problem
    PCONST=10000
    out = constant.benchmark_function(13)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = round(x[3])
    y2 = round(x[4])
    fx = -5.357854 * x1 ** 2 - 0.835689 * y1 * x3 - 37.29329 * y1 + 40792.141
    a = [85.334407, 0.0056858, 0.0006262, 0.0022053, 80.51249, 0.0071317, 0.0029955, 0.0021813, 9.300961, 0.0047026, 0.0012547, 0.0019085]
    gx = zeros(g)
    gx[0] = a[0] + a[1] * y2 * x3 + a[2] * y1 * x2 - a[3] * y1 * y1 * x3 - 92
    gx[1] = a[4] + a[5] * y2 * x3 + a[6] * y1 * x2 + a[7] * x1 ** 2 - 90 - 20
    gx[2] = a[8] + a[9] * y1 * x2 + a[10] * y1 * x1 + a[11] * x1 * x2 - 20 - 5
    hx = 0
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2)
    return PHI


def F6(x):
    #Stepped Cantilever Beam Design
    
#     xmin60=[1,1,1,1,1,30,30,30,30,30]
#     xmax60=[5,5,5,5,5,65,65,65,65,65]
    PCONST=10000
    L=100
    P=50000
    E=2*107
    fx=(x[0]* x[5]+x[1]*x[6]+x[2]*x[7]+x[3]*x[8]+x[4]*x[9])*L
    gx = zeros(11)
    gx[0]= ((600*P)/(x[4]*x[9]**2))-14000;
    gx[1]= ((6*P*2*L)/(x[3]*x[8]**2))-14000;
    gx[2]= ((6*P*3*L)/(x[2]*x[7]**2))-14000;
    gx[3]= ((6*P*4*L)/(x[1]*x[6]**2))-14000;
    gx[4]= ((6*P*5*L)/(x[0]*x[5]**2))-14000;
    gx[5]= (((P*L^3)/(3*E))*(125/L))-2.7;
    gx[6]=(x[9]/x[4])-20;
    gx[7]=(x[8]/x[3])-20;
    gx[8]=(x[7]/x[2])-20;
    gx[9]=(x[6]/x[1])-20;
    gx[10]=(x[5]/x[0])-20
    gx[gx<0] = 0
    PHI = fx + PCONST* (gx[0]** 2 + gx[1]** 2 + gx[2]** 2 + gx[3]** 2+ gx[4]** 2+ gx[5]** 2+ gx[6]** 2+ gx[7]** 2+ gx[8]** 2+ gx[9]** 2+ gx[10]** 2 )
    return PHI


# def F10(x):
#     # Gear train design Problem
#     x=round(x)
#     term1=1/6.931; 
#     #fit=((1/6.931)- floor (x(1))*floor (x(2))/floor (x(3))*floor (x(4)))^2;
#     term2=(x[0]*x[1])/(x[2]*x[3]);
#     PHI = (term1-term2)** 2
#     g = 0;
#     h = 0;
#     return PHI

def F10(x):
    # Gear train design Problem
    #x=round(x)
    term1=1/6.931; 
    #fit=((1/6.931)- floor (x(1))*floor (x(2))/floor (x(3))*floor (x(4)))^2;
    term2=(x[0]*x[1])/(x[2]*x[3]);

    #term2=(x[2]*x[1])/(x[0]*x[3]);
    PHI = (term1-term2)** 2
    g = 0;
    h = 0;
    return PHI

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", 1, 1, 1,17],
        "F2": ["F2", 1, 1, 1,18],
        "F3": ["F3", 1, 1, 1,20],
        "F4": ["F4", 1, 1, 1, 19],
        "F5": ["F5", 1, 1, 1, 15],
        "F6": ["F6", 1, 1, 1, 6],
        "F7": ["F7", 1, 1, 1, 21],
        "F8": ["F8", 1, 1, 1, 25],
        "F9": ["F9", 1, 1, 1, 13],
        "F10": ["F10",1, 1, 1, 10],
        "F11": ["F11", -100, 100, 30],
        "F12": ["F12", -100, 100, 30],
        "F13": ["F13", -100, 100, 30],
        "F14": ["F14", -100, 100, 30],
        "F15": ["F15", -100, 100, 30],
    }
    return param.get(a, "nothing")
