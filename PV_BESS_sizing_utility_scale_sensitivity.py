#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:22:15 2026

@author: pm.deoliveiradejes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis: Varying PV Installation Size
"""
import numpy as np
import numpy_financial as npf
import gurobipy as gp
from gurobipy import GRB, quicksum
import re
import csv
import matplotlib.pyplot as plt

# --- Data Loading Function ---
def read_inc(path):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.lstrip().startswith("t"):
                    continue
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 2:
                    hour  = int(parts[0][1:])           
                    value = float(parts[1])
                    values[hour] = value
    except FileNotFoundError:
        return {t: 0.1 for t in range(1, 8761)}
    return values 

# --- Parameters & Paths ---
paths = {
    'Plu': 'PluDataCenter.inc',
    'lambda': 'lambdaColombia_localtime.inc',
    'psi': 'psiColombia_localtime.inc',
    'Ppvu': 'PpvuCordobaERAS20052023_localtime.inc',
    "periodo": "periodo.inc",
}

series = {name: read_inc(route) for name, route in paths.items()}
T = range(1, 8761)
data = {t: {'lambda': series['lambda'].get(t, 0.0), 'Plu': series['Plu'].get(t, 0.0),
            'psi': series['psi'].get(t, 0.0), 'Ppvu': series['Ppvu'].get(t, 0.0)} for t in T}
periodo = {t: int(series["periodo"].get(t, 6)) for t in T}

er = 1 
kappa = [0, 0, 0, 0, 0, 0, 0] 
Plinst = 0
Rmax = 1000
Area = 2.4
eta = 0.2094
eff_c = 0.9381
eff_d = 0.9381
DoD = 0.9
PmaxF = 100000
BoP = 0
Sc = 1.32
OaMpv = 12.5
OaMbess = 5.9
CAPEX_pv = 388
CAPEX_BESS = 185
CAPEX_inverter = 48
i = 7.7/100
n = 20
e = 2.5/100
ir = (i-e)/(1+e)
crf = (i*(i+1)**n)/((i+1)**n-1)
crfe = (1+e)*(ir*(ir+1)**n)/((ir+1)**n-1)

# --- Model Initialization ---
m = gp.Model('SensitivitySizing')
m.setParam('OutputFlag', 0) # Quiet mode for loop efficiency

# Variables
AvG = m.addVar(name='Avg', lb=0)
Savings = m.addVar(name='Savings', lb=-GRB.INFINITY)
OPEX = m.addVar(name='OPEX', lb=0)
CapacityP = m.addVar(name='Capacity', lb=0)
CapacityP0 = m.addVar(name='Capacity0', lb=0)
npv_var = m.addVar(name='npv', lb=-GRB.INFINITY)
Benefit = m.addVar(name='Benefit', lb=-GRB.INFINITY)
OPEX0 = m.addVar(name='OPEX0', lb=-GRB.INFINITY)
Eb = m.addVar(name='Eb', lb=0)
Eb0 = m.addVar(name='Eb0', lb=0)
Es = m.addVar(name='Es', lb=0)
wpvmx = m.addVar(name='wpvmx', lb=-GRB.INFINITY)
wpv = m.addVar(name='wpv', lb=-GRB.INFINITY)
wclipping = m.addVar(name='wclipping', lb=-GRB.INFINITY)
Wb = m.addVar(name='Wb', lb=-GRB.INFINITY)
Ws = m.addVar(name='Ws', lb=-GRB.INFINITY)
Wl = m.addVar(name='Wl', lb=-GRB.INFINITY)
Wd = m.addVar(name='Wd', lb=-GRB.INFINITY)
Wc = m.addVar(name='Wc', lb=-GRB.INFINITY)
CashFlow_var = m.addVar(name='CashFlow', lb=-GRB.INFINITY)
Investment0 = m.addVar(name='Investment', lb=0)
PinverterBESS = m.addVar(name='PinverterBESS', lb=0)
PinverterPV = m.addVar(name='PinverterPV', lb=0)
Ppvinst = m.addVar(name='Ppvinst', lb=0) # This will be fixed in the loop
SOC0 = m.addVar(name='SOC0', lb=0)
nx = m.addVar(name='nx', lb=0)
C = m.addVar(name='C', lb=0)

Pbmax = {p: m.addVar(lb=0, name=f'PbmaxP{p}') for p in range(1, 7)}
SOC = m.addVars(T, lb=0, name='SOC')
Ppv = m.addVars(T, lb=0, name='Ppv')
Ppvmx = m.addVars(T, lb=0, name='Ppvmx')
Pd = m.addVars(T, lb=0, name='Pd')
Pc = m.addVars(T, lb=0, name='Pc')
Pb = m.addVars(T, lb=0, name='Pb')
Ps = m.addVars(T, lb=0, name='Ps')
w1 = m.addVars(T, vtype=GRB.BINARY, name='w1')
w3 = m.addVars(T, vtype=GRB.BINARY, name='w3')

# Basic Constraints
m.addConstrs((Pbmax[periodo[t]] >= Pb[t] for t in T), "res_Pbmax")
m.addConstr(Pbmax[6] == PmaxF)
for p in range(1, 6): m.addConstr(Pbmax[p+1] >= Pbmax[p])
for t in T:
    m.addConstr(Pd[t] + Pb[t] + Ppv[t] == Pc[t] + Ps[t] + Plinst * data[t]['Plu'])
    m.addConstr(Ppvmx[t] == Ppvinst * data[t]['Ppvu']) 
    if t == 1:
        m.addConstr(SOC[t] == SOC0 + Pc[t]*eff_c - Pd[t]/eff_d)
    else:
        m.addConstr(SOC[t] == SOC[t-1] + Pc[t]*eff_c - Pd[t]/eff_d)
    m.addConstr(Ppv[t] <= Ppvmx[t])
    m.addConstr(Pc[t] <= PinverterBESS * w1[t])
    m.addConstr(Pd[t] <= PinverterBESS * (1-w1[t]))
    m.addConstr(Pb[t] <= PmaxF * w3[t])
    m.addConstr(Ps[t] <= PmaxF * (1-w3[t]))
    m.addConstr(SOC[t] <= ((1-DoD)/2 + DoD)*C)
    m.addConstr(SOC[t] >= ((1-DoD)/2)*C)
m.addConstr(SOC0 <= ((1-DoD)/2 + DoD)*C)
m.addConstr(SOC0 >= ((1-DoD)/2)*C)

# Financial and Energy Constraints
m.addConstr(Es == er*(quicksum(data[t]['lambda'] * Ps[t] for t in T)))
m.addConstr(Eb == er*(quicksum((data[t]['lambda'] + data[t]['psi']) * Pb[t] for t in T)))
m.addConstr(Eb0 == er*(quicksum((data[t]['lambda'] + data[t]['psi']) * Plinst * data[t]['Plu'] for t in T)))
m.addConstr(CapacityP == quicksum(kappa[p]*Pbmax[p] for p in range(1, 7)))
m.addConstr(CapacityP0 == sum(kappa[p]*PmaxF for p in range(1, 7)))
m.addConstr(OPEX0 == Eb0 + CapacityP0)
m.addConstr(OPEX == CapacityP + Eb + OaMpv*Ppvinst + OaMbess*C)
m.addConstr(Savings == OPEX0 - OPEX)
m.addConstr(Benefit == Es - OPEX)
m.addConstr(CashFlow_var == Es + OPEX0 - OPEX)
m.addConstr(Investment0 == BoP + Sc * (CAPEX_pv*Ppvinst + CAPEX_BESS*C + CAPEX_inverter*(PinverterBESS+PinverterPV)))
m.addConstr(npv_var == CashFlow_var/(crfe) - Investment0)
m.addConstr(wpv == quicksum(Ppv[t] for t in T))
m.addConstr(wpvmx == quicksum(Ppvmx[t] for t in T))
m.addConstr(Wb == quicksum(Pb[t] for t in T))
m.addConstr(Ws == quicksum(Ps[t] for t in T))
m.addConstr(Wc == quicksum(Pc[t] for t in T))
m.addConstr(Wd == quicksum(Pd[t] for t in T))
m.addConstr(Wl == sum(Plinst * data[t]['Plu'] for t in T))
m.addConstr(wclipping == wpvmx - wpv)
m.addConstr(PinverterBESS <= C*2.0)
m.addConstr(PinverterBESS >= C*0.1)
m.addConstr(nx == 1000*Ppvinst/(Rmax*eta*Area))
m.addConstr(Wb == 0.00*Wl) 
m.addConstr(AvG == quicksum(data[t]['lambda'] for t in T)/8760) 
m.addConstr(PinverterPV == Plinst + PmaxF + PinverterBESS) 

m.setObjective(npv_var, GRB.MAXIMIZE)

# --- Sensitivity Loop ---
results = []
pv_range = range(100000, 1000001, 25000)

print(f"Starting Sensitivity Analysis for {len(pv_range)} iterations...")

for pv_val in pv_range:
    # Fix the Ppvinst variable for this iteration
    Ppvinst.lb = pv_val
    Ppvinst.ub = pv_val
    
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        inv = Investment0.X
        cf = CashFlow_var.X
        
        # Financial Calcs
        irr_val = npf.irr([-inv] + [cf]*n) * 100 if inv > 0 else 0
        try:
            pbt_val = np.log((cf) / (cf - i * inv)) / np.log(1 + i) if (cf - i * inv) > 0 else 0
        except:
            pbt_val = 0
            
        # Store: [Ppvinst, NPV, CAPEX, OPEX, IRR, ..., ..., PBT]
        # Filling col 6 and 7 with empty strings to respect your "column 8: PBT" request
        results.append([pv_val, npv_var.X, inv, OPEX.X, irr_val, "", "", pbt_val])
        print(f"PV Size: {pv_val} | NPV: {npv_var.X:,.2f}")
    else:
        print(f"PV Size: {pv_val} | No optimal solution found.")

# --- Save to CSV ---
header = ["Ppvinst", "NPV", "CAPEX", "OPEX", "IRR", "Empty1", "Empty2", "PBT"]
with open('sensitivity_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

print("\nDone! Results saved to sensitivity_results.csv")



# ... (Previous sensitivity loop code here) ...
# Ensure LCOE is calculated inside the loop:
# LCOEgross = 1000 * (inv + (OaMpv*pv_val + OaMbess*C.X)/crfe) / (Ws.X/crf) if Ws.X > 0 else 0
# results.append([pv_val, npv_var.X, inv, OPEX.X, irr_val, lcoe_val, pbt_val])

# --- Extract data for plotting ---
pv_axis = [r[0] for r in results]
npv_axis = [r[1] for r in results]
irr_axis = [r[4] for r in results]
lcoe_axis = [r[5] for r in results] # Ensure index matches your storage
pbt_axis = [r[7] for r in results]

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sensitivity Analysis: PV Installation Size ($P_{pvinst}$)', fontsize=16, fontweight='bold')

# Figure 1: NPV vs Ppvinst
axs[0, 0].plot(pv_axis, npv_axis, marker='o', color='b', linestyle='-')
axs[0, 0].set_title('Figure 1: NPV vs Ppvinst')
axs[0, 0].set_xlabel('PV Installed (kWdc)')
axs[0, 0].set_ylabel('Net Present Value (USD)')
axs[0, 0].grid(True, alpha=0.3)

# Figure 2: IRR vs Ppvinst
axs[0, 1].plot(pv_axis, irr_axis, marker='s', color='g', linestyle='-')
axs[0, 1].set_title('Figure 2: IRR vs Ppvinst')
axs[0, 1].set_xlabel('PV Installed (kWdc)')
axs[0, 1].set_ylabel('Internal Rate of Return (%)')
axs[0, 1].grid(True, alpha=0.3)

# Figure 3: PBT vs Ppvinst
axs[1, 0].plot(pv_axis, pbt_axis, marker='^', color='r', linestyle='-')
axs[1, 0].set_title('Figure 3: PBT vs Ppvinst')
axs[1, 0].set_xlabel('PV Installed (kWdc)')
axs[1, 0].set_ylabel('Payback Time (Years)')
axs[1, 0].grid(True, alpha=0.3)

# Figure 4: LCOE vs Ppvinst
axs[1, 1].plot(pv_axis, lcoe_axis, marker='d', color='m', linestyle='-')
axs[1, 1].set_title('Figure 4: Gross LCOE vs Ppvinst')
axs[1, 1].set_xlabel('PV Installed (kWdc)')
axs[1, 1].set_ylabel('LCOE (USD/MWh)')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()