###########################################################################################
#
#  logpot.py
#     Initialise a logarithmic potential
#     Useful for following Binney & Tremaine chapter 3
#
# 09 Jul 2021: Introduction
#
#
'''

logpot (part of exptool.models)
    Implementation of a planar logarithmic potential

'''




class LogPot():
    '''
    Planar Logarithmic Potential

    
    L = LogPot(v0=1.,q=0.8,rscl=0.1)
    print(L.get_pot(-1,-1))
    '''
    def __init__(self,rscl=1.,q=1.0,v0=1.):
        self.rscl2 = rscl*rscl
        self.q    = q
        self.v02   = v0*v0

    def get_pot(self,x,y):
        """B&T eq. 3.103
        
        G=M=1
        """
        logval = self.rscl2 + x*x + y*y/self.q/self.q
        return 0.5*self.v02 * np.log(logval)

    def get_xforce(self,x,y):
      """differentiate in Wolfram alpha
      
      https://www.wolframalpha.com/input/?i=d%2Fdx+0.5*a%5E2+*+ln%28b%5E2+%2B+x%5E2+%2B+y%5E2%2Fq%5E2%29
      """
      yscl2 = y*y/self.q/self.q
      return -(self.v02*x)/(self.rscl2 + yscl2 + x*x)
    
    def get_yforce(self,x,y):
      """differentiate in Wolfram alpha

      https://www.wolframalpha.com/input/?i=d%2Fdy+0.5*a%5E2+*+ln%28b%5E2+%2B+x%5E2+%2B+y%5E2%2Fq%5E2%29
      """
      return -(self.v02*y)/(self.q*self.q*(self.rscl2+x*x) + y*y)

    def get_cartesian_forces_array(self,arr):
      x = arr[0]
      y = arr[1]
      
      xforce = self.get_xforce(x,y)
      yforce = self.get_yforce(x,y)

      return np.array([xforce,yforce,0.])


