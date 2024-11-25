import numpy as np
import sys
sys.path.append('..')
from Code.fixed_income_derivatives_E2024 import *
# Problem 2 a-f
L_3M, L_6M = [0.01570161, 0.01980204]
pi_af = np.array([102.33689177,104.80430234,105.1615306,105.6581905,104.02899992,101.82604116, 1, 1, 1])

#2.a)
T = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
C = np.zeros((9, 9))
C[0,:] = [1, 1, 1, 1, 101, 0, 0, 0, 0]
C[1,:] = [2.5, 0, 2.5, 0, 102.5, 0, 0, 0, 0]
C[2,:] = [2.5, 0, 2.5, 0, 2.5, 0, 102.5, 0, 0]
C[3,:] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 101.5, 0, 0]
C[4,:] = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 101.25]
C[5,:] = [3, 0, 0, 0, 3, 0, 0, 0, 103]
C[6,:] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
C[7,:] = [0, 1 + L_3M/4, 0, 0, 0, 0, 0, 0, 0]
C[8,:] = [0, 0, 1 + L_6M/2, 0, 0, 0, 0, 0, 0]

#2.b)
ZCB = np.linalg.solve(C, pi_af)

#2.c)
forward_rates = forward_rates_from_zcb_prices(T, ZCB)
print("Forward rates: ", forward_rates)

#2.d)

principal = 100
PV_float = principal

#2.e)
par_swap_rate = (ZCB[0] - ZCB[8]) / np.sum(ZCB[2::2]*0.5)
print("Par swap rate: ", par_swap_rate)

# Problem 2 g-i
p_0330_gi, p_0630_gi = 0.99699147, 0.99088748
pi_gi = np.array([101.37241234,102.33995192,102.66601781,104.16399942,102.75471174,98.79916103])
rs_gi = -0.1161878302683732

#2.g)
T = [0, 2/12, 5/12, 8/12, 11/12, 14/12, 17/12, 20/12, 23/12]
L_3M *= 100
L_6M *= 100
par_swap_rate *= 100
C = np.zeros((10, 9))
C[0,:] = [0, 1, 1, 1, 101, 0, 0, 0, 0]
C[1,:] = [0, 0, 2.5, 0, 102.5, 0, 0, 0, 0]
C[2,:] = [0, 0, 2.5, 0, 2.5, 0, 102.5, 0, 0]
C[3,:] = [0, 1.5, 1.5, 1.5, 1.5, 1.5, 101.5, 0, 0]
C[4,:] = [0, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 101.25]
C[5,:] = [0, 0, 0, 0, 3, 0, 0, 0, 103]
C[6,:] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
C[7,:] = [0, 1, 0, 0, 0, 0, 0, 0, 0]
C[8,:] = [0, 0, 1, 0, 0, 0, 0, 0, 0]
C[9,:] = [0, -100 - L_3M/4, par_swap_rate/2, 0, par_swap_rate/2, 0, par_swap_rate/2, 0, 100 + par_swap_rate/2]

pi_gi = np.append(pi_gi, [1, p_0330_gi, p_0630_gi, rs_gi])

#2.h)
ZCB_gi = (np.linalg.inv(C.T @ C) @ C.T) @ pi_gi
ZCB_gi = np.linalg.lstsq(C, pi_gi)
print("ZCB_gi: ", ZCB_gi)

#2.i)
# ZCB is strictly positive, so arbitrage free
# Problem 2 j-k
p_0330_jk, p_0630_jk = 0.99391543, 0.98379379
pi_jk = np.array([100.00015573,100.95055325,100.77535024,100.26763545,100.48419302,96.56064083])
rs_jk = -2.04869321

#2.j)

pi_jk = np.append(pi_jk, [1, p_0330_jk, p_0630_jk, rs_jk])

ZCB_jk = np.linalg.lstsq(C, pi_jk, rcond=None)[0]
print("ZCB_jk: ", ZCB_jk)

#2.k)
C_ZCB = [0, 0, 0, 0, 0, 1, 0, 0, 0]
h_replica = np.linalg.inv(C) @ C_ZCB
print("h_replica: ", h_replica)