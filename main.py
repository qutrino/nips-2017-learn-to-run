from __future__ import print_function
import logging


from mpi4py import MPI
from osim.env.run import RunEnv
#import time
#import gym
from wrappers import ObsWrapper
from loops import master_loop, slave_loop

comm = MPI.COMM_WORLD   
size = comm.size        
rank = comm.rank  

if rank==1:
    visible = True
else:
    visible = False
    
env = RunEnv(visualize=visible, max_obstacles=10)  
env.seed(rank) 
#env = OsimWrapper(env)
env = ObsWrapper(env)
#ENV_NAME = 'Pendulum-v0'
#env = gym.make(ENV_NAME)
#env = env.unwrapped
 
if comm.rank == 0:
    master_loop(env)
else:
    slave_loop(env)

