#!/usr/bin/python
import random
import time
import numpy as np
import pylab as plt

__author__ = 'Guo Li'
__email__ = 'leeguoo@hotmail.com'
__date__ = 'Feb 22, 2017'

#Boltzmann constant (eV/K)
kB = 8.6173303E-5

class BinaryMC(object):
    """
    A class to simulation the formation of binary thin film on a substrate
    from a drop of particles.
    """
    def __init__(self, param, ngrid=100, nblocks=150, BAratio=0.3):
        """
        Initialize the simulation.
        """
        self.param = param
        self.ngrid = ngrid
        self.nblocks = nblocks
        self.ratio = BAratio
        self.ground, self.blocs = self.get_start()

    def get_start(self):
        """
        Constuct a square drop of particles.
        """
        nblocks = self.nblocks
        ngrid = self.ngrid
        ratio = self.ratio

        ground = np.zeros(shape=(ngrid/2,ngrid))
        nsqrt = int(np.ceil(np.sqrt(nblocks)))
        nmod = nblocks%nsqrt
        
        blocs = []
        for i in range(nblocks/nsqrt):
            for j in range((ngrid-nsqrt)/2,(ngrid+nsqrt)/2):
                if random.uniform(0,1) > ratio:
                    ground[i,j]= 1
                else:
                    ground[i,j]=-1
                blocs.append([i,j])
        if nmod != 0:
            i = nblocks/nsqrt
            for j in range((ngrid-nmod)/2,(ngrid+nmod)/2):
                if random.uniform(0,1) > ratio:
                    ground[i,j]= 1
                else:
                    ground[i,j]=-1
                blocs.append([i,j])
        return ground, blocs

    def move(self):
        nb = random.randint(0,self.nblocks-1)      #select a block
        y, x = self.blocs[nb]                      #get the location of the block
        dx = random.choice([-1,0,0,1])             #select a direction along x
        if dx == 0:                                #select a direction along y
            dy = random.choice([-1,1])
        else:
            dy = 0
        x1 = self.periodic(x,dx)
        y1 = y+dy

        #diffusion
        if self.ground[y1][x1]==0 and dx!=0 and self.ground[y+1][x]==0:
            #find the highest y at that x
            while True:
                if y1==0 or self.ground[y1-1][x1]!=0:
                    break
                else:
                    y1 -= 1
            #print self.Probability(x,y,x1,y1,dx), "diffusion"
            if random.uniform(0,1)<=self.Probability(x,y,x1,y1,dx):
                self.ground[y1][x1]=self.ground[y][x] #move the block
                self.ground[y][x]=0
                self.blocs[nb] = [y1,x1]        #update the block list
        #exchange
        elif self.ground[y1][x1]!=0 and y1>=0 and self.ground[y1][x1]!=self.ground[y][x]:
            for i, val in enumerate(self.blocs):
                if val == [y1,x1]:
                    nb1 = i
                    break
            #print self.Probability(x,y,x1,y1,dx), "exchange"
            if random.uniform(0,1)<=self.Probability(x,y,x1,y1,dx):
                #if y!=y1:
                #    print self.ground[y][x],self.ground[y1][x1]
                temp = self.ground[y1][x1]
                self.ground[y1][x1]=self.ground[y][x] #move the block
                self.ground[y][x]=temp
                self.blocs[nb] = [y1,x1]        #update the block list
                self.blocs[nb1] = [y,x]


    def periodic(self,x,dx):
        if x+dx < 0:
            return x+dx+self.ngrid
        elif x+dx > self.ngrid-1:
            return x+dx-self.ngrid
        else:
            return x+dx


    def bindingE(self,x,y):
        V = 0
        if self.ground[y][x]==1: #Type-A block
            #up
            if self.ground[y+1][x]==1:
                V += self.param['EAA']
            elif self.ground[y+1][x]==-1:
                V += self.param['EAB']
            #down
            if y==0:
                V += self.param['EAS']
            elif self.ground[y-1][x]==1:
                V += self.param['EAA']
            elif self.ground[y-1][x]==-1:
                V += self.param['EAB']
            #left
            if self.ground[y][self.periodic(x,-1)]==1:
                V += self.param['EAA']
            elif self.ground[y][self.periodic(x,-1)]==-1:
                V += self.param['EAB']
            #right
            if self.ground[y][self.periodic(x,1)]==1:
                V += self.param['EAA']
            elif self.ground[y][self.periodic(x,1)]==-1:
                V += self.param['EAB']
        elif self.ground[y][x]==-1: #Type-B block
            #up
            if self.ground[y+1][x]==1:
                V += self.param['EAB']
            elif self.ground[y+1][x]==-1:
                V += self.param['EBB']
            #down
            if y==0:
                V += self.param['EBS']
            elif self.ground[y-1][x]==1:
                V += self.param['EAB']
            elif self.ground[y-1][x]==-1:
                V += self.param['EBB']
            #left
            if self.ground[y][self.periodic(x,-1)]==1:
                V += self.param['EAB']
            elif self.ground[y][self.periodic(x,-1)]==-1:
                V += self.param['EBB']
            #right
            if self.ground[y][self.periodic(x,1)]==1:
                V += self.param['EAB']
            elif self.ground[y][self.periodic(x,1)]==-1:
                V += self.param['EBB']
        else:
            print "Error"
        return V


    def Probability(self,x,y,x1,y1,dx):
        """
        Calculate the probability of the movement at room temperature.

        Args:
            x, y (int): the current location of the block
            x1, y1 (int): the new location
        
        Returns:
            Relative probability (float): at room temperature (300K), and
            the probability of the fatest movement is set as 1.
        """
        V = 0
        #diffusion
        if self.ground[y1][x1]==0: 
            if self.ground[y][x]==1: # A-type block
                #diffusion barrier
                if y==0:
                    V += self.param['EAS']*self.param['AlphaA']
                elif self.ground[y-1][x]==1:
                    V += self.param['EAA']*self.param['AlphaA']
                elif self.ground[y-1][x]==-1:
                    V += self.param['EAB']*self.param['AlphaA']
                #detaching barrier
                if self.ground[y][self.periodic(x,-dx)]==1:
                    V += self.param['EAA']
                elif self.ground[y][self.periodic(x,-dx)]==-1:
                    V += self.param['EAB']
            elif self.ground[y][x]==-1: #B-type block
                #diffusion barrier
                if y==0:
                    V += self.param['EBS']*self.param['AlphaB']
                elif self.ground[y-1][x]==1:
                    V += self.param['EAB']*self.param['AlphaB']
                elif self.ground[y-1][x]==-1:
                    V += self.param['EBB']*self.param['AlphaB']
                #detaching barrier
                if self.ground[y][self.periodic(x,-dx)]==1:
                    V += self.param['EAB']
                elif self.ground[y][self.periodic(x,-dx)]==-1:
                    V += self.param['EBB']
        else: #exchange
            V = self.bindingE(x,y)
            V = V+self.bindingE(x1,y1)
            V = V*self.param['AlphaC']
        #find the lowest energy barrier
        Vref = min(param['AlphaA'],param['AlphaB'])*min(param['EAA'],param['EBB'],param['EAB'],param['EAS'],param['EBS'])
        #print V, np.exp((Vref-V)/300/kB)
        return np.exp((Vref-V)/300/kB)

    def plot(self,name='tmp.png'):
        plt.imshow(MC.ground,cmap='bwr',origin='uper')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(name)

    def simulate(self,nsteps=10000000,nfigs=20):
        self.plot('0.png')
        nspan = nsteps/nfigs
        for i in range(nsteps):
            self.move()
            if (i+1)%nspan==0:
                self.plot(str((i+1)/nspan)+'.png')

param = {"EAA":0.12,   #A-A interaction
         "EBB":0.04,    #B-B interaction
         "EAB":0.05,    #A-B interaction
         "EAS":0.2,    #A-surface interaction
         "EBS":0.2,    #B-surface interaction
         "AlphaA":1.0,   #diffusion coefficient
         "AlphaB":1.0,
         "AlphaC":0.2    #exchange coefficient
        }

start = time.time()
MC = BinaryMC(param=param,ngrid=50,nblocks=70,BAratio=0.3)
MC.simulate(nsteps=80000000,nfigs=10)
print time.time()-start
