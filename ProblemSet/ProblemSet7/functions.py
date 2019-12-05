import numpy as np

def state(lb,ub,sale_q):
    '''
    ------------------------------------------------------------------------
    Create Grid for State Space
    ------------------------------------------------------------------------
    lb      = scalar, lower bound of initial Q grid
    ub      = scalar, upper bound of initial Q grid
    sale_q  = integer, number of grid points in sales state space
    q_grid  = vector, sale_q x 1 vector of sales grid points
    ------------------------------------------------------------------------
    '''
    q_grid = np.linspace(lb,ub,sale_q)
    return q_grid

def profit(sale_q,q_grid,epsilon,alpha,Cv,Cf,b):
    '''
    ------------------------------------------------------------------------
    Create grid of current profit
    ------------------------------------------------------------------------
    S        = matrix, current consumption (s=q-q')
    Pi       = matrix, current period profit value for all possible
               choices of q and q' (rows are q, columns q')
    ------------------------------------------------------------------------
    '''

    S = np.zeros((sale_q, sale_q))
    for i in range(sale_q): # loop over q
        for j in range(sale_q): # loop over q'
            S[i, j] = q_grid[i] - q_grid[j] # note that if q'>q, sales negative
# replace 0 and negative sales with a tiny value

    S[S<=0] = 1e-15

    Pi = (1/(epsilon*alpha))*(S**2)-(b/(epsilon*alpha)+Cv)*S-Cf

    Pi[S<0] = -9999999

    return Pi


def VFI(VFtol,VFdist,VFmaxiter,sale_q,Pi,beta):
    '''
    ------------------------------------------------------------------------
    Value Function Iteration
    ------------------------------------------------------------------------
    VFtol     = scalar, tolerance required for value function to converge
    VFdist    = scalar, distance between last two value functions
    VFmaxiter = integer, maximum number of iterations for value function
    V         = vector, the value functions at each iteration
    Vmat      = matrix, the value for each possible combination of q and q'
    Vstore    = matrix, stores V at each iteration
    VFiter    = integer, current iteration number
    TV        = vector, the value function after applying the Bellman operator
    PF        = vector, indicies of choices of q' for all q
    VF        = vector, the "true" value function
    ------------------------------------------------------------------------
    '''

    V = np.zeros(sale_q) # initial guess at value function
    Vmat = np.zeros((sale_q, sale_q)) # initialize Vmat matrix
    Vstore = np.zeros((sale_q, VFmaxiter)) #initialize Vstore array
    VFiter = 1
    while VFdist > VFtol and VFiter < VFmaxiter:
        for i in range(sale_q): # loop over q
            for j in range(sale_q): # loop over q'
                Vmat[i, j] = Pi[i, j] + beta * V[j]

        Vstore[:, VFiter] = V.reshape(sale_q,) # store value function
        TV = Vmat.max(1) # apply max operator to Vmat (to get V(q))
        PF = np.argmax(Vmat, axis=1)
        VFdist = (np.absolute(V - TV)).max()  # check distance
        V = TV
        VFiter += 1

    VF = V # solution to the functional equation
    return PF,VF
