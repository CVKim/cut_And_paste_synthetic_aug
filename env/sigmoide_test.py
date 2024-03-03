
import numpy as np
import torch
import torchvision
import math

def sigmoide(x, deff=False):
    if deff : 
        return sigmoide(x)*(1-sigmoide(x))
    else :
        return 1 / (1 + math.exp(-x))
    

print(1 / (1 - 0.7311))

print(sigmoide(1))
print(sigmoide(0))
print(sigmoide(0.5))

print(math.exp(0))