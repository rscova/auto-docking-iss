from flask import Flask, send_file, send_from_directory
import os
from flask_socketio import SocketIO, emit

import threading
from threading import Event

import time
import random
import math
import numpy as np


class Env:
    def __init__(self,dt=1.0):
        self.dt = dt
        self.dv = 0.0597
        self.max_state = [30.0,3.0,3.0,3.0,3.0,3.0]

        self.done = 0
        self.state = [0.0] * 12
        self.velocity_next = [0.0] * 6

        self.actions_counts = [0] * 6
        self.new_action = False

        self.done_reward = 0.0

    def run(self):
        pass
    
    def compute_reward(self):
        reward_x = 0.0
        reward_y = 0.0
        reward_z = 0.0
        reward_roll = 0.0
        reward_pitch = 0.0
        reward_yaw = 0.0

        # if self.state[0] > 1.0:
        #     reward_x = max(-(self.state[0] - 1) / 29, -1)  # 0 + (-1-0)*(x-1)/(30-1)
        # else:
        #     reward_x =  min(-(self.state[0] - 1), 1)# 0 + (1.0-0)*(x-1)/(0-1)

        if abs(self.state[1]) > 0.2:
            reward_y = max(-(abs(self.state[1]) - 0.2) / 2.8, -1)# 0 + (-1-0)*(x-0.2)/(3-0.2)
        else:
            reward_y = min(-(abs(self.state[1]) - 0.2) / 0.2, 1) # 0 + (1-0)*(x-0.2)/(0-0.2)

        if abs(self.state[2]) > 0.2:
            reward_z = max(-(abs(self.state[2]) - 0.2) / 2.8, -1) # 0 + (-1-0)*(x-0.2)/(3-0.2)
        else:
            reward_z = min(-(abs(self.state[2]) - 0.2) / 0.2, 1) # 0 + (1-0)*(x-0.2)/(0-0.2)
    
        if abs(self.state[6]) > 0.2:
            reward_roll = max(-(abs(self.state[6]) - 0.2) / 2.8, -1) # 0 + (-1-0)*(x-0.2)/(3-0.2)
        else:
            reward_roll = min(-(abs(self.state[6]) - 0.2) / 0.2, 1) # 0 + (1-0)*(x-0.2)/(0-0.2)

        if abs(self.state[7]) > 0.2:
            reward_pitch = max(-(abs(self.state[7]) - 0.2) / 2.8, -1) # 0 + (-1-0)*(x-0.2)/(3-0.2)
        else:
            reward_pitch = min(-(abs(self.state[7]) - 0.2) / 0.2, 1) # 0 + (1-0)*(x-0.2)/(0-0.2)

        if abs(self.state[8]) > 0.2:
            reward_yaw = max(-(abs(self.state[8]) - 0.2) / 2.8, -1) # 0 + (-1-0)*(x-0.2)/(3-0.2)
        else:
            reward_yaw = min(-(abs(self.state[8]) - 0.2) / 0.2, 1) # 0 + (1-0)*(x-0.2)/(0-0.2)
        
        return reward_x + reward_y + reward_z + reward_roll + reward_pitch + reward_yaw


    def reset(self):
        self.done = 0
        self.velocity_next = [0.0] * 6

        self.actions_counts = [0] * 6
        self.new_action = False
        self.done_reward = 0.0

        # self.state = [15.0,0.0,0.0,
        #               0.0,0.0,0.0,
        #               0.0,0.0,0.0,
        #               0.0,0.0,0.0]
        
        self.state = [random.uniform(15, 20),
                      random.uniform(-2, 2),
                      random.uniform(-2, 2),
                      0.0,0.0,0.0,
                      random.uniform(-2, 2),
                      random.uniform(-2, 2),
                      random.uniform(-2, 2),
                      0.0,0.0,0.0]

        return self.state
    
    def update_done(self):
        self.done = 0
        self.done_reward = 0

        if (self.state[0] >= self.max_state[0] or abs(self.state[1]) >= self.max_state[1] or abs(self.state[2]) >= self.max_state[2] or
            abs(self.state[6]) >= self.max_state[3] or abs(self.state[7]) >= self.max_state[4] or abs(self.state[8]) >= self.max_state[5]
            ):
            self.done = 1
            self.done_reward = -200.0
    
        # if self.state[0] < 0.0:
        #     if (abs(self.state[1]) < 0.2 and abs(self.state[2]) < 0.2 and 
        #         abs(self.state[6]) < 0.2 and abs(self.state[7]) < 0.2 and abs(self.state[8]) < 0.2
        #         ):
        #         self.done = 1
        #         self.done_reward = 200.0
        #     else:
        #         self.done = 1
        #         self.done_reward = 0.0
        
    def rotation_matrix(self,roll, pitch, yaw):
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        return R

    # Función para actualizar la posición y la orientación
    def update_position_orientation(self,position, orientation, linear_velocity, angular_velocity):
        roll, pitch, yaw = orientation * math.pi / 180
        R = self.rotation_matrix(roll, pitch, yaw)
        
        # Actualizar posición
        delta_position = np.dot(R, linear_velocity) * self.dt
        position += delta_position
        
        # Actualizar orientación
        delta_orientation = angular_velocity * self.dt
        orientation += delta_orientation

        # if position[0] < 0.0:
        #     position[0] = 0.0
        
        return position, orientation

    def update_state(self,action):
        #Update state with previous velocities
        self.state[3] = self.velocity_next[0] # vx
        self.state[4] = self.velocity_next[1] # vy
        self.state[5] = self.velocity_next[2] # vz
        self.state[-3] = self.velocity_next[-3] # wx
        self.state[-2] = self.velocity_next[-2] # wy
        self.state[-1] = self.velocity_next[-1] # wz

        position, orientation = self.update_position_orientation(np.array(self.state[:3]),
                                         np.array(self.state[6:9]),
                                         np.array(self.state[3:6]),
                                         np.array(self.state[-3:]))

        self.state[:3] = position.tolist()
        self.state[6:9] = orientation.tolist()

        #Update next velocities with commands
        if self.new_action:
            #Linear
            if action == 0:
                self.velocity_next[0] += 0.0#self.dv
            elif action == 1:
                self.velocity_next[0] -= 0.0#self.dv
            elif action == 2:
                self.velocity_next[1] -= self.dv
            elif action == 3:
                self.velocity_next[1] += self.dv
            elif action == 4:
                self.velocity_next[2] += self.dv
            elif action == 5:
                self.velocity_next[2] -= self.dv
            #Angular
            elif action == 6:
                self.velocity_next[3] += self.dv
            elif action == 7:
                self.velocity_next[3] -= self.dv
            elif action == 8:
                self.velocity_next[4] += self.dv
            elif action == 9:
                self.velocity_next[4] -= self.dv
            elif action == 10:
                self.velocity_next[5] += self.dv
            elif action == 11:
                self.velocity_next[5] -= self.dv
    
    def saturate_actions(self,action):
        self.new_action = False
        if action == 0:
            self.actions_counts[0] += 1
            if self.actions_counts[0] <= 2:
                self.new_action = True
            else:
                self.actions_counts[0] = 2
        elif action == 1:
            self.actions_counts[0] -= 1
            if self.actions_counts[0] >= -2:
                self.new_action = True
            else:
                self.actions_counts[0] = -2
        elif action == 2:
            self.actions_counts[1] += 1
            if self.actions_counts[1] <= 1:
                self.new_action = True
            else:
                self.actions_counts[1] = 1
        elif action == 3:
            self.actions_counts[1] -= 1
            if self.actions_counts[1] >= -1:
                self.new_action = True
            else:
                self.actions_counts[1] = -1
        elif action == 4:
            self.actions_counts[2] += 1
            if self.actions_counts[2] <= 1:
                self.new_action = True
            else:
                self.actions_counts[2] = 1
        elif action == 5:
            self.actions_counts[2] -= 1
            if self.actions_counts[2] >= -1:
                self.new_action = True
            else:
                self.actions_counts[2] = -1
        
        elif action == 6:
            self.actions_counts[3] += 1
            if self.actions_counts[3] <= 1:
                self.new_action = True
            else:
                self.actions_counts[3] = 1
        elif action == 7:
            self.actions_counts[3] -= 1
            if self.actions_counts[3] >= -1:
                self.new_action = True
            else:
                self.actions_counts[3] = -1
        elif action == 8:
            self.actions_counts[4] += 1
            if self.actions_counts[4] <= 1:
                self.new_action = True
            else:
                self.actions_counts[4] = 1
        elif action == 9:
            self.actions_counts[4] -= 1
            if self.actions_counts[4] >= -1:
                self.new_action = True
            else:
                self.actions_counts[4] = -1
        elif action == 10:
            self.actions_counts[5] += 1
            if self.actions_counts[5] <= 1:
                self.new_action = True
            else:
                self.actions_counts[5] = 1
        elif action == 11:
            self.actions_counts[5] -= 1
            if self.actions_counts[5] >= -1:
                self.new_action = True
            else:
                self.actions_counts[5] = -1  
        

    def step(self,action:int):
        self.saturate_actions(action)
        self.update_state(action)
        self.update_done()
        
        reward = self.compute_reward() + self.done_reward

        return self.state, reward, self.done
    
    def close(self):
        pass