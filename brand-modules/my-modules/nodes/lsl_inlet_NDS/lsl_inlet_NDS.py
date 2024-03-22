#!/usr/bin/env python
import gc
import logging
import sys
from struct import unpack
from pylsl import StreamInlet, resolve_byprop, local_clock
import threading
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from brand import BRANDNode

class NeuralDataStreamInlet(BRANDNode):
    def __init__(self, desired_streams):
        super().__init__()
        self.streams_config = desired_streams
        self.inlets = self._setup_inlets()
        self.running = True
        self.data = {stream_name: [] for stream_name in desired_streams}

    #Create LSL inlet to received data from Neural Data Simulator
    def _setup_inlets(self):
        inlets = {}
        for stream_name, config in self.streams_config.items():
            streams = resolve_byprop('name', config['stream_name'])
            if streams:
                inlet = StreamInlet(streams[0])
                inlets[stream_name] = inlet
            else:
                logging.info(f"No streams found for {stream_name}")
        return inlets

    def receive_data(self, stream_name):
        if stream_name in self.inlets:
            inlet = self.inlets[stream_name]
            logging.info(f"Receiving data for {stream_name}. Press Enter to stop.\n")
            while self.running:
                sample, send_timestamp = inlet.pull_sample(timeout=0.5)
                if sample:
                    #Latency calculation
                    received_time = local_clock()# Use LSL's local_clock for the current time
                    latency = received_time - send_timestamp# Calculate latency (in seconds)
                    self.data[stream_name].append((send_timestamp, received_time, latency, sample))  # Store all data together
                    # Preparing the data for sending to Redis as JSON
                    data_to_send = json.dumps({'timestamp': send_timestamp, 'sample': sample.tolist() if isinstance(sample, np.ndarray) else sample})
                    # Sending the data to Redis
                    self.r.xadd(stream_name, {'data': data_to_send})
        else:
            raise ValueError(f"Stream {stream_name} not configured or not found.")
        
    def save_data_to_csv(self):
        logging.info("Saving data to CSV...")
        
        # Create a session-specific folder name based on the current date and time
        session_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_dir = os.path.join('..','DATA_test', session_folder)
        
        # Ensure the session-specific DATA directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            # Set the directory permissions to 755 (rwx for owner, rx for group and others)
            # This line sets the permissions right after creating the directory
            os.chmod(data_dir, 0o755)

        for stream_name, samples in self.data.items():
            # Modify the filename to include the session-specific DATA directory
            filename = os.path.join(data_dir, f"{stream_name}.csv")
            try:
                with open(filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    #Define CSV headers including new fields
                    csvwriter.writerow(['Send Timestamp', 'Received Timestamp', 'Latency', 'Sample'])
                    for send_timestamp, received_time, latency, sample in samples:
                        # Save a row for each received sample including new data
                        csvwriter.writerow([send_timestamp, received_time, latency, sample])
                logging.info(f"Data for {stream_name} saved to {filename}")
            except Exception as e:
                logging.info(f"Error saving {stream_name} data to CSV: {e}")

    def get_stream_sample_values(self, stream_name):
        #Return only the sample values for a specific stream, excluding timestamps.
        if stream_name in self.data:
            return [sample for timestamp, sample in self.data[stream_name]]
        else:
            raise ValueError(f"Stream {stream_name} not found.")
        
    def terminate(self, sig, frame):
        self.running = False
        self.save_data_to_csv()
        logging.info("Stopping data reception...")
        super().terminate(sig, frame)      

    def run(self):
        #Create threads for each stream
        threads = []
        for stream_name in self.streams_config:
            thread = threading.Thread(target=self.receive_data, args=(stream_name,), daemon=True)
            threads.append(thread)
            thread.start()

        try:
            while any(thread.is_alive() for thread in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received, stopping threads.")
            self.terminate()
        
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gc.disable()

    desired_streams = {
        #The information regarding LSL streams from Neural Data Simulator can be found here:
        #
        #Raw, LPF, Spike Events, and Spike Rates:
        #https://agencyenterprise.github.io/neural-data-simulator/configuring.html
        #
        #Position, Behavior Velocity, Decoder:
        #https://agencyenterprise.github.io/neural-data-simulator/tasks.html#using-the-gui
        'Raw': {'stream_name': 'NDS-RawData', 'stream_type': 'Ephys'},
        'LFP': {'stream_name': 'NDS-LFPData', 'stream_type': 'LFP'},
        'SpikeEvents': {'stream_name': 'NDS-SpikeEvents', 'stream_type': 'SpikeEvents'},
        'Position': {'stream_name': 'NDS-TaskWindow', 'stream_type': 'Position'},
        'BehaviorVelocity': {'stream_name': 'NDS-Behavior', 'stream_type': 'CursorVelocity'},
        'Decoder': {'stream_name': 'NDS-Decoder', 'stream_type': 'Decoder'},
        'SpikeRates': {'stream_name': 'NDS-SpikeRates', 'stream_type': 'SpikingRates'}
    }

    stream_inlet = NeuralDataStreamInlet(desired_streams)
    stream_inlet.run()

    gc.collect()
