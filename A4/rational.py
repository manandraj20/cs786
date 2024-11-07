import numpy as np
from random import shuffle

# Utility functions:

class dLocalMAP:
    """
    See Anderson (1990, 1991)
    'Categories' renamed 'clusters' to avoid confusion.
    Discrete version.
    
    Stimulus format is a list of integers from 0 to n-1 where n is the number
    of possible features (e.g. [1,0,1])
    
    args: c, alphas
    """
    
    def __init__(self, args):
        self.partition = [[]]
        self.c, self.alpha = args
        self.alpha0 = sum(self.alpha.T)
        print(self.alpha.T)
        print(self.alpha0)
        self.N = 0
    
    def probClustVal(self, k, i, val):
        """Find P(j|k)"""
        cj = len([x for x in self.partition[k] if x[i]==val])
        nk = len(self.partition[k])
        return (cj + self.alpha[i][val])/(nk + self.alpha0[i])
    
    def condclusterprob(self, stim, k):
        """Find P(F|k)"""
        pjks = []
        for i in range(len(stim)):
            cj = len([x for x in self.partition[k] if x[i]==stim[i]])
            nk = len(self.partition[k])
            pjks.append( (cj + self.alpha[i][stim[i]])/(nk + self.alpha0[i]) )
        return np.prod( pjks )
        
    
    def posterior(self, stim):
        """Find P(k|F) for each cluster"""
        pk = np.zeros( len(self.partition) )
        pFk = np.zeros( len(self.partition) )
        
        # existing clusters:
        for k in range(len(self.partition)):
            pk[k] = self.c * len(self.partition[k])/ ((1-self.c) + self.c * self.N)
            if len(self.partition[k])==0: # case of new cluster
                pk[k] = (1-self.c) / (( 1-self.c ) + self.c * self.N)
            pFk[k] = self.condclusterprob( stim, k)
        
        # put it together
        pkF = (pk*pFk) # / sum( pk*pFk )
        
        return pkF
    
    def stimulate(self, stim):
        """Argmax of P(k|F) + P(0|F)"""
        winner = np.argmax( self.posterior(stim) )
        print("Stim: ", stim)
        print("Partition: ", self.partition)
        print(self.posterior(stim))
        
        if len(self.partition[winner]) == 0:
            self.partition.append( [] )
        self.partition[winner].append(stim)
        
        self.N += 1
    
    def query(self, stimulus):
        """Queried value should be -1."""
        qdim = -1
        for i in range(len(stimulus)):
            if stimulus[i] < 0:
                if qdim != -1:
                    raise Exception("ERROR: Multiple dimensions queried.")
                qdim = i
        
        self.N = sum([len(x) for x in self.partition])
        
        pkF = self.posterior(stimulus)
        print("pkF: ", pkF[:-1])
        pkF = pkF[:-1] / sum(pkF[:-1]) # eliminate `new cluster' prob
        
        pjF = np.array( [sum( [ pkF[k] * self.probClustVal(k, qdim, j) \
                for k in range(len(self.partition)-1)] ) 
                for j in range(len( self.alpha[qdim] ))] )
        
        return pjF / sum(pjF)

def testlocalmapD():
    """
    Tests the Anderson's ratinal model using the Medin & Schaffer (1978) data.
    
    This script will print out the probability that each item belongs to each
    of the existing clusters or to a new cluster, and the model assign it to
    the most likely cluster. To see that the model is working correctly, you
    can follow along with Anderson (1991), which steps through in the same way.
    """
    stims = [[1, 1, 1, 1, 1], # Medin & Schaffer (1978)
             [1, 0, 1, 0, 1], 
             [1, 0, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 1, 0, 1, 1],  
             [0, 1, 0, 0, 0]]
    # These are the classic Shepard Type II and Type IV datasets.
    # Uncomment the one you want to try out; you might want to uncomment
    # shuffling the stims too if you don't care about order.
    #stims = [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 1, 1]] # Type IV
    stims = [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]] # Type II
    
    for _ in range(1):  # Use range() instead of xrange()
        model = dLocalMAP([.5, np.ones((len(stims[0]),2))])
        
        shuffle(stims)
        for s in stims:
            model.stimulate(s)
        print(model.partition)
        
        print("Prob vals for 0,0,0,0,?", model.query( [0]*(len(stims[0])-1) + [-1] ))

def main():
    testlocalmapD()

if __name__ == '__main__':
    main()