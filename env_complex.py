from flask import Flask, send_file, send_from_directory
import os
from flask_socketio import SocketIO, emit

import threading
from threading import Event

import time
import random
import math

class Env:
    def __init__(self,port=5000):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.stop_event = Event()

        self.reset_cmd = 0
        self.max_episode_time_seconds = 200

        self.sim_data = [0.0,0.0,0.0]
        self.sim_past_data = [0.0,0.0,0.0]

        self.state_curr = [0.0,0.0,0.0]
        self.state_past = [0.0,0.0,0.0]

        self.stopped = False
        self.new_action = False
        self.done = 0
        self.reward = 0.0

        self.port = port
        self.init_dist = 0

        self.dt = 0.01

        self.max_state = [0,0,0,0,0,0]

        self.derivative_timer = time.time()

        self.actions = [0,0,0,0,0,0]
        self.actions_counts = [0,0,0]

    def configure_routes(self):
        @self.app.route('/')
        def index():
            return send_file('iss-sim.spacex.com/index.html')

        @self.app.route('/<path:filename>')
        def static_files(filename):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.join(root_dir,'iss-sim.spacex.com')
            return send_from_directory(os.path.join(root_dir, ''), filename)

    def error_data_socket(self):
        @self.socketio.on('error_data')
        def error_data(data):
            self.sim_past_data = self.sim_data
            self.sim_data = [float(value) for value in data]

            if (self.reset_cmd):
                self.socketio.emit('datos_hacia_cliente', [0,0,0,0,0,0,1]) # 0 actions, (1) Reset
                self.new_action = False
                self.reset_cmd = 0
            elif (self.stopped == False): #Continue while not stopped
                self.socketio.emit('datos_hacia_cliente', [0,0,0,0,0,0,0])
            else: 
                if (self.new_action):
                    data_to_send = self.actions.copy()
                    self.socketio.emit('datos_hacia_cliente', data_to_send + [0]) # (0) No reset
                    self.new_action = False
                else:
                    self.socketio.emit('datos_hacia_cliente', [0,0,0,0,0,0,0])
    
    def done_data_socket(self):
        @self.socketio.on('done_data')
        def done_data(done):
            self.done = done[0]

            self.socketio.emit('datos_hacia_cliente',[1])

    def run_flask_server(self):
        self.socketio.run(self.app, debug=True, host="127.0.0.1", port=self.port, use_reloader=False,allow_unsafe_werkzeug=True)
        self.stop_event.set()  # Set the event when Flask exits

    def run(self):
        self.configure_routes()
        self.error_data_socket()
        self.done_data_socket()
        thread = threading.Thread(target=self.run_flask_server) 
        thread.daemon = True
        thread.start()
    
    def compute_reward(self):
        reward_x = 0.0
        reward_y = 0.0
        reward_z = 0.0

        if abs(self.state_curr[0]) > 1.0:
            reward_x = (-self.state_curr[0] + 1) / 29
        else:
            reward_x = 0.5 * (-self.state_curr[0] + 1)

        if abs(self.state_curr[1]) > 0.2:
            reward_y = -(abs(self.state_curr[1]) - 0.2) / 2.8
        else:
            reward_y = -(abs(self.state_curr[1]) - 0.2) / 0.2

        if abs(self.state_curr[2]) > 0.2:
            reward_z = -(abs(self.state_curr[2]) - 0.2) / 2.8
        else:
            reward_z = -(abs(self.state_curr[2]) - 0.2) / 0.2
        

        # reward_dist = -math.sqrt( ((self.state_curr[0]/20.0)**2+(self.state_curr[1]/2.0)**2+(self.state_curr[2]/2.0)**2) / 3.0)
        # reward_x = 0.0
        # reward_y = 0.0
        # reward_z = 0.0

        # if abs(self.state_curr[1]) <= 0.2:
        #     reward_y = -2.5 * abs(self.state_curr[1]) + 0.5 # max +0.5 reward
        # if abs(self.state_curr[2]) <= 0.2:
        #     reward_z = -2.5 * abs(self.state_curr[2]) + 0.5 # max +0.5 reward
        
        # if self.state_curr[0] - self.state_past[0] < 0.0:
        #     reward_x = 0.1
        # else:
        #     reward_x= 0.1

        # print (reward_x,reward_y,reward_z)

        return reward_x + reward_y + reward_z

    def wait_until_stopped(self,sleep_time = 0.1):
        count = 0
        while count < 20:
            if self.is_stopped():
                count +=1
            else: 
                count = 0
            time.sleep(sleep_time)
        
        # print("Stopped")
        self.stopped = True
        return True
    
    def is_stopped(self):
        return False if sum(self.sim_data[i] - self.sim_past_data[i] for i,_ in enumerate(self.sim_data)) != 0 else True

    def reset(self):
        self.reset_cmd = 1
        self.stopped = False
        print("Reset")
        self.wait_until_stopped()
        self.derivative_timer = time.time() - 1
        self.new_action = False
        self.actions_counts = [0,0,0]
        state = self.get_state().copy()
        # self.init_dist = math.sqrt(self.state_curr[0]**2+self.state_curr[1]**2+self.state_curr[2]**2)

        return self.get_state().copy()
    
    def get_state(self):
        self.state_past = self.state_curr.copy()
        self.state_curr = self.sim_data.copy()

        state = self.state_curr.copy()
        curr_time = time.time()
        self.dt = curr_time - self.derivative_timer

        state.append( (self.state_curr[0]-self.state_past[0]) / self.dt )
        state.append( (self.state_curr[1]-self.state_past[1]) / self.dt )
        state.append( (self.state_curr[2]-self.state_past[2]) / self.dt )

        # for s_curr,s_past in zip(self.state_curr,self.state_past):
        #     state.append( (s_curr-s_past) / self.dt )

        self.derivative_timer = curr_time
        return state
    
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

    def step(self,action:int):
        self.actions = [0,0,0,0,0,0]
        if action < 6:
            self.actions[action] = 1

        self.saturate_actions(action)

        state = self.get_state()
        
        reward = self.compute_reward()

        if (state[0] >= 30.0 or abs(state[1]) >= 3.0 or abs(state[2]) >= 3.0):
            return state, reward-200, -1
        else:
            return state, reward, self.done
    
    def close(self):
        pass