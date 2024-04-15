from random import uniform
import numpy as np
from src.misc import S
from src.misc import dS
from src.misc import flatten

class Network():

    def __init__(self, layout: list, eta = 0.1):
        self.layout = layout
        self.eta = eta

        self.nlayers = []
        for s in self.layout:
            self.nlayers.append(np.random.rand(s))

        self.errors = np.zeros(layout[-1])

        self.wlayers = []
        for l in range(len(self.layout)-1):
            size_wlayer = (self.nlayers[l+1].size, self.nlayers[l].size)
            self.wlayers.append(np.random.rand(*size_wlayer))

    def read(self, picture):
        #self.picture = picture
        self.picture = np.array(flatten(picture))
        self.nlayers[0] = (self.picture-min(self.picture))/(max(self.picture)-min(self.picture))

    def propagate(self):
        for l in range(1, len(self.layout)):
            self.nlayers[l] = S(np.matmul(self.wlayers[l-1], self.nlayers[l-1]))

    def calculate_errors(self, label: int):
        self.errors[label] = 1
        self.errors = self.errors - self.nlayers[-1] 

        return np.sum(np.square(self.errors))/2

    def backprop(self):
        #self.wlayers[-1] = self.wlayers[-1] - self.eta*dS(self.wlayers[-1].sum(axis=0))
        #print(self.wlayers[-1][0])
        #print(self.wlayers[-1][0].sum())
        #print(self.wlayers[-1].sum(axis=1))
        #print(delta)
        d_epsilon = -self.errors*dS(self.wlayers[-1].sum(axis=1))
        self.wlayers[-1] = np.add(self.wlayers[-1], -self.eta*d_epsilon*self.nlayers[-1])
        #print(self.nlayers[-1].shape)
        #for l in range(-2, 1):
        #    print(l)
