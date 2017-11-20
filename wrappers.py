# reference
# https://github.com/ctmakro/stanford-osrl/blob/master/observation_processor.py


DELTA_T = 0.01

V_FACTOR = 10.0
VR_FACTOR = 2.0
OBS_MAX = 2.0
OBS_MIN = -0.7

class ObsPlain(object):
    def __init__(self, observation):
        o = observation
        self.r_pelvis = o[0]
        self.x_pelvis = o[1]
        self.y_pelvis = o[2]
        
        self.vr_pelvis = o[3] / VR_FACTOR  # 3
        self.vx_pelvis = o[4] / V_FACTOR
        self.vy_pelvis = o[5] / V_FACTOR
        
        self.r_ankle_1 = o[6]  # 6
        self.r_ankle_2 = o[7]
        
        self.r_knee_1 = o[8]
        self.r_knee_2 = o[9]
        
        self.r_hip_1 = o[10]
        self.r_hip_2 = o[11]
        
        self.vr_ankle_1 = o[12] / VR_FACTOR  # 12
        self.vr_ankle_2 = o[13] / VR_FACTOR
        
        self.vr_knee_1 = o[14] / VR_FACTOR
        self.vr_knee_2 = o[15] / VR_FACTOR
        
        self.vr_hip_1 = o[16] / VR_FACTOR
        self.vr_hip_2 = o[17] / VR_FACTOR
        
        self.x_cm = o[18]  # 18
        self.y_cm = o[19]
        
        self.vx_cm = o[20] / V_FACTOR
        self.vy_cm = o[21] / V_FACTOR
        
        self.x_head = o[22] 
        self.y_head = o[23]
        
        # self.x_pelvis = o[0]
        # self.y_pelvis = o[0]
        
        self.x_torso = o[26] 
        self.y_torso = o[27]
        self.x_toe_1 = o[28] 
        self.y_toe_1 = o[29]
        self.x_toe_2 = o[30] 
        self.y_toe_2 = o[31]
        self.x_talus_1 = o[32] 
        self.y_talus_1 = o[33]
        self.x_talus_2 = o[34] 
        self.y_talus_2 = o[35]
        
        self.s_psoas_1 = o[36]
        self.s_psoas_2 = o[37]
        
        self.x_obstacle = o[38] 
        self.y_obstacle = o[39]
        self.r_obstacle = o[40]
        
        
        
        #print("x0: {}, x: {}, y: {}, r: {}".format(o[38]+o[1], o[38], o[39], o[40]))
        #self.x_pelvis = 0
        

class ObsProcessor(object):
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.prev = None
        self.stored_obstacle_x_abs = -1.0 # abs value
        self.stored_obstacle_y = 0
        self.stored_obstacle_r = 0
      
    def apply(self, observation):
        
        cur = ObsPlain(observation)
        
        if self.prev == None:
            self.prev = cur
     
        obstacle_x_abs = cur.x_obstacle + cur.x_pelvis
        prev_obstacle_x_abs = self.prev.x_obstacle + self.prev.x_pelvis
        
        if abs(obstacle_x_abs-prev_obstacle_x_abs) > 0.0001 and prev_obstacle_x_abs < 99.0 : # different obstacle
            self.stored_obstacle_x_abs = prev_obstacle_x_abs
            self.stored_obstacle_y = self.prev.y_obstacle
            self.stored_obstacle_r = self.prev.r_obstacle
        
        res = []
        
        res.append(cur.r_pelvis)
        # res.append(cur.x_pelvis)
        res.append(cur.y_pelvis)

        res.append(cur.vr_pelvis)  # o[3] #3
        res.append(cur.vx_pelvis)  # o[4]
        res.append(cur.vy_pelvis)  # o[5]
        
        res.append(cur.r_ankle_1)  # o[6] #6
        res.append(cur.r_ankle_2)  # o[7]
        
        res.append(cur.r_knee_1)  # o[8]
        res.append(cur.r_knee_2)  # o[9]
        
        res.append(cur.r_hip_1)  # o[10]
        res.append(cur.r_hip_2)  # o[11]
        
        res.append(cur.vr_ankle_1)  # o[12] #12
        res.append(cur.vr_ankle_2)  # o[13]
        
        res.append(cur.vr_knee_1)  # o[14]
        res.append(cur.vr_knee_2)  # o[15]
        
        res.append(cur.vr_hip_1)  # o[16]
        res.append(cur.vr_hip_2)  # o[17]
        
        res.append(cur.x_cm - cur.x_pelvis)  # o[18] #18
        res.append(cur.y_cm)  # o[19]
        
        res.append(cur.vx_cm)  # o[20]
        res.append(cur.vy_cm)  # o[21]
        
        res.append(cur.x_head - cur.x_pelvis)  # o[22]
        res.append(cur.y_head)  # o[23]
        
        # cur.x_pelvis # o[0]
        # cur.y_pelvis # o[0]
        
        res.append(cur.x_torso - cur.x_pelvis)  # o[26]
        res.append(cur.y_torso)  # o[27]
        res.append(cur.x_toe_1 - cur.x_pelvis)  # o[28]
        res.append(cur.y_toe_1)  # o[29]
        res.append(cur.x_toe_2 - cur.x_pelvis)  # o[30]
        res.append(cur.y_toe_2)  # o[31]
        res.append(cur.x_talus_1 - cur.x_pelvis)  # o[32]
        res.append(cur.y_talus_1)  # o[33]
        res.append(cur.x_talus_2 - cur.x_pelvis)  # o[34]
        res.append(cur.y_talus_2)  # o[35]
        
        res.append(cur.s_psoas_1)  # o[36]
        res.append(cur.s_psoas_2)  # o[37]
        
        stored_x_obstable = self.stored_obstacle_x_abs-cur.x_pelvis
        #print("-obs1: {}, {}, {}".format(stored_x_obstable, self.stored_obstacle_y, self.stored_obstacle_r))
        if stored_x_obstable < OBS_MIN or stored_x_obstable > OBS_MAX:
            res.append(-1)  
            res.append(0.) 
            res.append(0.)
        else:    
            res.append(stored_x_obstable)  
            res.append(self.stored_obstacle_y) 
            res.append(self.stored_obstacle_r) 
        
        #print(" obs1: {}, {}, {}".format(res[-3], res[-2], res[-1]))
        
        #print("-obs2: {}, {}, {}".format(cur.x_obstacle, cur.y_obstacle, cur.r_obstacle))
        if cur.x_obstacle < OBS_MIN or cur.x_obstacle > OBS_MAX:
            res.append(-1)  
            res.append(0.) 
            res.append(0.)
        else:    
            res.append(cur.x_obstacle)  # o[38]
            res.append(cur.y_obstacle)  # o[39]
            res.append(cur.r_obstacle)  # o[40]
        #print(" obs2: {}, {}, {}".format(res[-3], res[-2], res[-1]))
        #print( "abs1 ", self.stored_obstacle_x_abs , self.stored_obstacle_x_abs-cur.x_pelvis, self.stored_obstacle_y, self.stored_obstacle_r) 
        #print( "abs2 ", obstacle_x_abs , cur.x_obstacle, cur.y_obstacle, cur.r_obstacle) 
        
        

        # add relative y
        res.append(cur.y_cm - cur.y_pelvis) 
        res.append(cur.y_head - cur.y_pelvis) 
        res.append(cur.y_torso - cur.y_pelvis)  
        res.append(cur.y_toe_1 - cur.y_pelvis)  
        res.append(cur.y_toe_2 - cur.y_pelvis)  
        res.append(cur.y_talus_1 - cur.y_pelvis)  
        res.append(cur.y_talus_2 - cur.y_pelvis)  
        #res.append(cur.y_obstacle- cur.y_pelvis)
        
        
        # add relative vx, vy
        res.append(cur.vx_cm - cur.vx_pelvis)
        res.append(cur.vy_cm - cur.vy_pelvis)
        
        vx_head = (cur.x_head - self.prev.x_head)/DELTA_T/V_FACTOR
        vy_head = (cur.y_head - self.prev.y_head)/DELTA_T/V_FACTOR
        
        vx_torso = (cur.x_torso - self.prev.x_torso)/DELTA_T/V_FACTOR
        vy_torso = (cur.y_torso - self.prev.y_torso)/DELTA_T/V_FACTOR
        
        vx_toe_1 = (cur.x_toe_1 - self.prev.x_toe_1)/DELTA_T/V_FACTOR
        vy_toe_1 = (cur.y_toe_1 - self.prev.y_toe_1)/DELTA_T/V_FACTOR
        
        vx_toe_2 = (cur.x_toe_2 - self.prev.x_toe_2)/DELTA_T/V_FACTOR
        vy_toe_2 = (cur.y_toe_2 - self.prev.y_toe_2)/DELTA_T/V_FACTOR
        
        vx_talus_1 = (cur.x_talus_1 - self.prev.x_talus_1)/DELTA_T/V_FACTOR
        vy_talus_1 = (cur.y_talus_1 - self.prev.y_talus_1)/DELTA_T/V_FACTOR
        
        vx_talus_2 = (cur.x_talus_2 - self.prev.x_talus_2)/DELTA_T/V_FACTOR
        vy_talus_2 = (cur.y_talus_2 - self.prev.y_talus_2)/DELTA_T/V_FACTOR
        
        
        res.append(vx_head)
        res.append(vy_head)
        res.append(vx_torso)
        res.append(vy_torso)
        
        res.append(vx_toe_1)
        res.append(vy_toe_1)
        res.append(vx_toe_2)
        res.append(vy_toe_2)
        
        res.append(vx_talus_1)
        res.append(vy_talus_1)
        res.append(vx_talus_2)
        res.append(vy_talus_2)
        
        res.append(vx_head- cur.vx_pelvis)
        res.append(vy_head- cur.vy_pelvis)
        res.append(vx_torso- cur.vx_pelvis)
        res.append(vy_torso- cur.vy_pelvis)
        
        res.append(vx_toe_1- cur.vx_pelvis)
        res.append(vy_toe_1- cur.vy_pelvis)
        res.append(vx_toe_2- cur.vx_pelvis)
        res.append(vy_toe_2- cur.vy_pelvis)
        
        res.append(vx_talus_1- cur.vx_pelvis)
        res.append(vy_talus_1- cur.vy_pelvis)
        res.append(vx_talus_2- cur.vx_pelvis)
        res.append(vy_talus_2- cur.vy_pelvis)    
        
        
        self.prev = cur
        
        return res
    
    def get_dim(self): 
        return 74
          
class ObsWrapper:
    def __init__(self,e):
        self.e = e
        self.obs_processor = ObsProcessor()
       
    def step(self, action):
        obs, reward, is_done, info = self.e.step(action)
        processed = self.obs_processor.apply(obs)
        return processed, reward, is_done, info

    def reset(self):
        self.obs_processor.clear()
        obs = self.e.reset(difficulty=2)
        processed = self.obs_processor.apply(obs)
        return processed
    
    def get_s_dim(self):
        return self.obs_processor.get_dim()
        
    def get_a_dim(self):
        return self.e.action_space.shape[0]   
    
    def get_a_high(self):
        return self.e.action_space.high[0]
        
    def get_a_low(self):
        return self.e.action_space.low[0]
    