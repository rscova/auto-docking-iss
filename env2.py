from flask import Flask, send_file, send_from_directory
import os
from flask_socketio import SocketIO, emit

import threading
from threading import Event

import time
import random
import math

class Env:
    def __init__(self):
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
            # self.reward = float(done[1])
            # if (self.current_steps <= self.max_steps):
            #     self.done = done[0]
            #     self.reward = float(done[1])
            # else:
            #     self.done = -1
            #     self.reward = -3000.0

            self.socketio.emit('datos_hacia_cliente',[1])

    def run_flask_server(self):
        self.socketio.run(self.app, debug=True, host="127.0.0.1", port=5000, use_reloader=False,allow_unsafe_werkzeug=True)
        self.stop_event.set()  # Set the event when Flask exits

    def run(self):
        self.configure_routes()
        self.error_data_socket()
        self.done_data_socket()
        thread = threading.Thread(target=self.run_flask_server) 
        thread.daemon = True
        thread.start()
    
    def compute_reward(self,state):
        reward = 0.0

        for i in range(3):
            if abs(self.state_curr[i]) - abs(self.state_past[i]) > 0:
                reward -= 0.3
            elif abs(self.state_curr[i]) - abs(self.state_past[i]) < 0:
                reward += 0.3
            elif abs(self.state_curr[i]) - abs(self.state_past[i]) == 0:
                reward += 0.05
    
        #reward += self.reward

        return reward

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
        self.state_curr = self.sim_data.copy()
        self.state_past = self.state_curr.copy()
        self.new_action = False
        self.actions_counts = [0,0,0]
        return self.get_state().copy()
    
    def get_state(self):
        self.state_past = self.state_curr.copy()
        self.state_curr = self.sim_data.copy()

        state = self.state_curr.copy()
        curr_time = time.time()
        self.dt = curr_time - self.derivative_timer

        for s_curr,s_past in zip(self.state_curr,self.state_past):
            state.append( ( abs(s_past)-abs(s_curr) ) / self.dt )

        self.derivative_timer = curr_time
        return state

    def step(self,action:int):
        self.new_action = False

        self.actions = [0,0,0,0,0,0]
        if action < 6:
            self.actions[action] = 1


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
            if self.actions_counts[1] <= 2:
                self.new_action = True
            else:
                self.actions_counts[1] = 2
        elif action == 3:
            self.actions_counts[1] -= 1
            if self.actions_counts[1] >= -2:
                self.new_action = True
            else:
                self.actions_counts[1] = -2
        elif action == 4:
            self.actions_counts[2] += 1
            if self.actions_counts[2] <= 2:
                self.new_action = True
            else:
                self.actions_counts[2] = 2
        elif action == 5:
            self.actions_counts[2] -= 1
            if self.actions_counts[2] >= -2:
                self.new_action = True
            else:
                self.actions_counts[2] = -2

        state = self.get_state()
        reward = self.compute_reward(state)

        return state, reward, self.done
    
    def close(self):
        pass