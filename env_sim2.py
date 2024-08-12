from flask import Flask, send_file, send_from_directory
import os
from flask_socketio import SocketIO, emit

import threading
from threading import Event

import time
import random
import math



class Env:
    def __init__(self,dt=1.0):
        self.dt = dt
        self.dv = 0.0597
        self.max_state = [30.0,3.0,3.0]

        self.done = 0
        self.state = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.velocity_next = [0.0,0.0,0.0]

        self.actions_counts = [0,0,0]
        self.new_action = False

        self.done_reward = 0.0
        self.steps = 0

    def run(self):
        pass
    
    def compute_reward(self):
        reward_x = 0.0
        reward_y = 0.0
        reward_z = 0.0
        reward_step = 0.0

        if abs(self.state[0]) > 5.0:
            reward_x = (-self.state[0] + 5) / 25
        else:
            reward_x = 2.0 * (-self.state[0] + 5)

        if abs(self.state[1]) > 0.2:
            reward_y = -(abs(self.state[1]) - 0.2) / 2.8
        else:
            reward_y = -(abs(self.state[1]) - 0.2) / 0.2

        if abs(self.state[2]) > 0.2:
            reward_z = -(abs(self.state[2]) - 0.2) / 2.8
        else:
            reward_z = -(abs(self.state[2]) - 0.2) / 0.2
    
        
        reward_step = 1 - self.steps / 200  # 1  + (0-1)*(x-0) / (100-0) 

        return reward_x + reward_y + reward_z + reward_step


    def reset(self):
        self.done = 0
        self.state = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.velocity_next = [0.0,0.0,0.0]

        self.actions_counts = [0,0,0]
        self.new_action = False
        self.done_reward = 0.0
        self.steps = 0
        
        self.state = [random.uniform(15, 20),
                      random.uniform(-2, 2),
                      random.uniform(-2, 2),
                      0.0,0.0,0.0]

        return self.state
    
    def update_done(self):
        self.done = 0
        self.done_reward = 0

        if (self.state[0] >= self.max_state[0] or abs(self.state[1]) >= self.max_state[1] or abs(self.state[2]) >= self.max_state[2]):
            self.done = 1
            self.done_reward = -200.0
    
        if self.state[0] < 0.0:
            if abs(self.state[1]) < 0.2 and abs(self.state[2]) < 0.2:
                self.done = 1
                self.done_reward = 200.0
            else:
                self.done = 1
                self.done_reward = 50.0
        

    def update_state(self,action):
        #Update state with previous velocities
        self.state[3] = self.velocity_next[0]
        self.state[4] = self.velocity_next[1]
        self.state[5] = self.velocity_next[2]

        self.state[0] = self.state[0] + self.state[3] * self.dt
        self.state[1] = self.state[1] + self.state[4] * self.dt
        self.state[2] = self.state[2] + self.state[5] * self.dt

        #Update next velocities with commands
        if self.new_action:
            if action == 0:
                self.velocity_next[0] += self.dv
            elif action == 1:
                self.velocity_next[0] -= self.dv
            elif action == 2:
                self.velocity_next[1] -= self.dv
            elif action == 3:
                self.velocity_next[1] += self.dv
            elif action == 4:
                self.velocity_next[2] += self.dv
            elif action == 5:
                self.velocity_next[2] -= self.dv
    
    def saturate_actions(self,action):
        self.new_action = False
        if action == 0:
            self.actions_counts[0] += 1
            if self.actions_counts[0] <= 1:
                self.new_action = True
            else:
                self.actions_counts[0] = 1
        elif action == 1:
            self.actions_counts[0] -= 1
            if self.actions_counts[0] >= -1:
                self.new_action = True
            else:
                self.actions_counts[0] = -1
        elif action == 1:
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

    def step(self,action:int):
        self.steps += 1
        self.saturate_actions(action)
        self.update_state(action)
        self.update_done()
        
        reward = self.compute_reward() + self.done_reward

        # if self.new_action:
        #     print(action,self.state,self.actions_counts)
        # else:
        #     print(7,self.state,self.actions_counts)


        return self.state, reward, self.done
    
    def close(self):
        pass