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
        # self.episode_init_time = time.time()

        self.sim_data = [0.0,0.0,0.0]#[0.0,0.0,0.0,0.0,0.0,0.0]
        self.sim_past_data = [0.0,0.0,0.0]#[0.0,0.0,0.0,0.0,0.0,0.0]

        self.state_curr = [0.0,0.0,0.0]#[0.0,0.0,0.0,0.0,0.0,0.0]
        self.state_past = [0.0,0.0,0.0]#[0.0,0.0,0.0,0.0,0.0,0.0]

        self.stopped = False
        self.new_action = False
        self.done = 0
        self.reward = 0.0

        self.dt = 0.01

        self.max_steps = 3000
        self.current_steps = 0

        self.max_state = [0,0,0,0,0,0]

        # self.derivative_timer = time.time()

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
            # print(data[-1])
            # if float(data[-1]) != 0.0:
            #     self.dt = float(data[-1]*1e-3)

            # data = data[:-1]
            self.sim_data = [float(value) for value in data]
            # print(self.sim_data)

            # self.stopped = self.is_stopped()

            # timeout = (time.time() - self.episode_init_time) >= self.max_episode_time_seconds

            if (self.reset_cmd):
                # print("Timeout or reset" ,self.reset_cmd, " ",self.sim_data[0])
                self.socketio.emit('datos_hacia_cliente', [0,0,0,0,0,0,1]) # 0 actions, (1) Reset
                # self.episode_init_time = time.time()
                self.new_action = False
                # self.done = -1
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
            # print ("Received done")
            # print(done)
            if (self.current_steps <= self.max_steps):
                self.done = done[0]
                self.reward = float(done[1])
            else:
                self.done = -1
                self.reward = -3000.0

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

        # self.stop_event.wait()
        # self.reset()
    
    def compute_reward(self,state):
        reward = 0.0

        for i in range(len(self.state_curr)):
            if abs(self.state_curr[i]) - abs(self.state_past[i]) > 0:
                reward -= 1
            elif abs(self.state_curr[i]) - abs(self.state_past[i]) < 0:
                reward += 0.33
    
        reward += self.reward
        # if self.done == -1:
        #     reward = self.reward - math.sqrt(sum([state[0]**2,state[1]**2,state[2]**2]))
        #     print(self.reward,reward)
        
        # elif self.done == 1:
        #     reward = self.reward
        #     print(self.reward,reward)
        # else:
        #     reward =  -math.sqrt(sum([state[0]**2,state[1]**2,state[2]**2])) / 40

        return reward

    def wait_until_stopped(self,sleep_time = 0.1):
        count = 0
        while count < 20:
            if self.is_stopped():
                count +=1
            else: 
                count = 0
            time.sleep(sleep_time)
            
        
        print("Stopped")
        self.stopped = True
        return True
    
    def is_stopped(self):
        return False if sum(self.sim_data[i] - self.sim_past_data[i] for i,_ in enumerate(self.sim_data)) != 0 else True

    def reset(self):
        self.reset_cmd = 1
        self.stopped = False
        print("Reset")
        self.wait_until_stopped()
        self.state_curr = self.sim_data.copy()
        self.state_past = self.state_curr.copy()
        # self.episode_init_time = time.time()
        self.new_action = False
        self.current_steps = 0
        # self.derivative_timer = time.time()
        return self.get_state().copy()
    
    def get_state(self):
        self.state_past = self.state_curr.copy()
        self.state_curr = self.sim_data.copy()

        # print(self.sim_data,self.sim_past_data)

        # state = self.state_curr.copy()
        # for s_curr,s_past in zip(self.state_curr,self.state_past):
        #     state.append( ( abs(s_past)-abs(s_curr) ) / self.dt )

        # print(self.dt)
        return self.state_curr

    def step(self,action:int):
        self.current_steps += 1

        self.actions = [0,0,0,0,0,0]
        if action < 6:
            self.actions[action] = 1
        self.new_action = True

        # curr_time = time.time()
        # self.dt = curr_time - self.derivative_timer
        # self.derivative_timer = curr_time
        state = self.get_state()
        reward = self.compute_reward(state)
        # print(state)
        # for i in range(6):
        #     if abs(state[i]) > abs(self.max_state[i]):
        #         self.max_state[i] = state[i]
        #         print(self.max_state)
        return state, reward, self.done
    
    def close(self):
        pass