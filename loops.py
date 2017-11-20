from __future__ import print_function

from datetime import datetime
import math
import time

from mpi4py import MPI
from osim.env.run import RunEnv


import numpy as np
import tensorflow as tf
import logging
from rl.random import OrnsteinUhlenbeckProcess

from model import DDPG

comm = MPI.COMM_WORLD   
size = comm.size        
rank = comm.rank  


#####################  MPI TAG  ####################
EPS_DATA = 0
OBS_DATA = 1
REQ_ACTION = 3
RSP_ACTION = 4

#####################  hyper parameters  ####################

MAX_EPISODES = 100000
MAX_EP_STEPS = 1000
LR_A = 0.00005  # learning rate for actor
LR_C = 0.0003  # learning rate for critic
GAMMA = 0.99  # reward discount 
TAU = 0.001  # soft replacement
MEMORY_CAPACITY = 1000000
LEARN_START = 10000
BATCH_SIZE = 128


def master_loop(env):
    
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fileHandler = logging.FileHandler('./log/test.log')
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)

    s_dim = env.get_s_dim()
    a_dim = env.get_a_dim()
    a_high = env.get_a_high()
    a_low = env.get_a_low()
    # print(a_bound)
    print("s_dim: {}, a_dim{}, a_high:{}, a_low:{}".format(s_dim, a_dim, a_high, a_low))
    ddpg = DDPG(a_dim, s_dim, a_high, a_low, lr_a=LR_A, lr_c=LR_C,
                gamma=GAMMA, tau=TAU, rpm_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE)
  
    status = MPI.Status()   
    start_time = time.time()
    reset_time = time.time()
    
    total_eps = 0
    total_step = 0
    
    n_step = 0
    n_eps = 0
    
    max_reward = -9999
    max_reward_rank = 0

    ddpg.load()
   
    while total_eps < MAX_EPISODES:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == REQ_ACTION:
            # action = env.action_space.sample()
            action = ddpg.choose_action(data)
            comm.send((action, total_eps, total_step), dest=source, tag=RSP_ACTION)

        elif tag == OBS_DATA:
            n_step += 1
            total_step += 1
            (s, a, r, s_, done, ep_reward, ep_step) = data
            is_done = 0.0;
            if done:
                is_done = 1.0

            ddpg.store_transition(s, a, r, s_, is_done)

            if ddpg.pointer > LEARN_START and total_step % 3 == 0:
                ddpg.learn()
            
            if done:
                total_eps += 1
                if ep_reward > max_reward:
                    max_reward = ep_reward
                    max_reward_rank = source
                
                s = "eps: {:>8}, worker: {:>3}, ep_reward:{:7.4f}, max:{:7.4f}/{:>3}, step:{:4}".format(
                    total_eps, source, ep_reward, max_reward, max_reward_rank, ep_step)        
                #print(s)
                logging.info(s)  
                       
                
                if total_eps % 500 == 0:
                    ddpg.save(total_eps)
                    interval = time.time() - reset_time
                    s = "# total_step: {:>8} ,total_eps: {:>6} eps/min: {:>6}, frame/sec: {:>6}".format(
                        total_step, total_eps, n_eps / interval * 60, n_step / interval)
                    #print(s)
                    logging.info(s)
                 
                    
                    n_step = 0
                    n_eps = 0
                    reset_time = time.time();
                

NOISE_DECAY = 0.9995
NOISE_MIN = 0.02

def slave_loop(env):
    np.random.seed(seed=rank)
    s = env.reset()  # env.reset_with_seed(rank)

    eps_count = 0
    step_count = 0
    ep_reward = 0.
    ep_step = 0

    ou_noise = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.e.action_space.shape[0])
    a_high = env.e.action_space.high[0]
    a_low = env.e.action_space.low[0]
    
    initial_noise_scale = 1.0  
    noise_decay = NOISE_DECAY  # 0.99
    global_n_eps = 0
    global_n_step = 0
    
    while True:
        #noise_scale = max(initial_noise_scale * noise_decay ** (global_n_eps), 0.002)
        noise_scale = max(initial_noise_scale * noise_decay ** (global_n_eps), NOISE_MIN)

        if rank == 1:
            noise_scale = 0
        
        comm.send(np.array(s), dest=0, tag=REQ_ACTION)
        (action, global_n_eps, global_n_step) = comm.recv(source=0, tag=RSP_ACTION)

        noise = ou_noise.sample()
        action = np.clip(action + noise * noise_scale, a_low, a_high)
  
        s_, reward, done, info = env.step(action)
        ep_reward += reward
        
        step_count += 1
        ep_step += 1
        
        if ep_step >= MAX_EP_STEPS:
            done = True
        
        obs_data = (np.array(s), action, reward, np.array(s_), done, ep_reward, ep_step)
        comm.send(obs_data, dest=0, tag=OBS_DATA)
            
        if done:            
            s_ = env.reset()
            ep_reward = 0
            eps_count += 1
            ep_step = 0
            ou_noise.reset_states()
            # print("eps: %d" % (eps_count,))     
        s = s_
