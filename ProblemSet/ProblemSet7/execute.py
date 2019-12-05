# Import necessary packages

import functions as func
import numpy as np
import matplotlib.pyplot as plt
# to print plots inline
# %matplotlib inline

# Set some basic parameter

sale_q = 400
beta = (1/1.1)
alpha = -0.8
Cv = 1
Cf = 2
epsilon = 1
b = 15

q_grid = func.state(1,400,sale_q)

Pi = func.profit(sale_q,q_grid,epsilon,alpha,Cv,Cf,b)

PF,VF = func.VFI(1e-8,7.0,1000,sale_q,Pi,beta)

'''
------------------------------------------------------------------------
Find sales and savings policy functions
------------------------------------------------------------------------
optQ  = vector, the optimal choice of q' for each q
optS  = vector, the optimal choice of S for each S
optP  = vector, the optimal choice of price
------------------------------------------------------------------------
'''
optQ = q_grid[PF] # tomorrow's optimal inventory size (savings function)
optS = q_grid - optQ # optimal sales
optP = (optS-b)/epsilon*alpha # optimal price, calculated by optimal sales

# Plot value function
plt.figure()
plt.scatter(q_grid[1:], VF[1:])
plt.xlabel('Quantity of products')
plt.ylabel('Value Function')
plt.title('Value Function - deterministic product sales')
plt.show()

#Plot optimal price rule as a function of inventory
plt.figure()
fig, ax = plt.subplots()
ax.plot(q_grid[3:], optP[3:], label='Prices')

legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Quantity of products')
plt.ylabel('Optimal Price')
plt.title('Policy Function, sales - deterministic product sales')
plt.show()
