#!/usr/bin/env python
import gc
import logging
import sys
from struct import unpack
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop, local_clock
import threading
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from brand import BRANDNode
import pdb
import redis
import pylsl

class RedisToLSL():
    def __init__(self):
        
        self.parameters = dict()
        
        #Update ip and port parameters
        self.parameters['host_ip'] = '10.157.174.27'
        self.parameters['host_port'] = 50000      
        self.host_ip = self.parameters['host_ip']
        self.host_port = self.parameters['host_port']
        self.input_stream_name = 'neural_data'
        self.characteristic_to_stream = b'spikes'
        self.window_time_points = 25

        # Create LSL StreamOutlet
        stream_name='Spikes'
        stream_type='SpikingRates'
        num_channels=137
        sampling_rate=100
        info = StreamInfo(name=stream_name, type=stream_type, channel_count=num_channels, nominal_srate=sampling_rate, channel_format='int16', source_id='random1234')
        self.outlet = StreamOutlet(info)
        
        super().__init__()
        self.running = True

    def received_redis_data(self):
        while(self.running):
             #Get data from Redis
             data = self.r.xread(streams={self.input_stream_name:"$"}, count=1, block=0)
             sample = np.frombuffer(data[0][1][0][1][self.characteristic_to_stream], dtype=np.int32)
             #Reshape and sum data to have size of 1xchannels
             sample = np.reshape(sample, (int(len(sample)/self.window_time_points), self.window_time_points))
             sample = np.sum(sample, axis = 1)
             #Push data to lsl outlet
             self.outlet.push_sample(sample)
        
    def terminate(self, sig, frame):
        self.running = False
        logging.info("Stopping data reception...")
        super().terminate(sig, frame)      

    def run(self):
        self.r = redis.Redis(host = self.host_ip, port = self.host_port, decode_responses=False)
        self.received_redis_data()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gc.disable()
    stream_redis_to_lsl = RedisToLSL()
    stream_redis_to_lsl.run()

    gc.collect()
