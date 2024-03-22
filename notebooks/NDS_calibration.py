# Load the graph
import json
import os
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from brand.timing import timespecs_to_timestamps, timevals_to_timestamps
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import redis
import yaml

DURATION = None  # seconds
GRAPH = 'sim_graph_NDS.yaml'
REDIS_IP = '127.0.0.1'
REDIS_PORT = 6379
test_dir = os.getcwd()

with open(os.path.join(test_dir, 'graphs', GRAPH), 'r') as f:
    graph = yaml.safe_load(f)

#Step 1
r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)

#Step 2 and 3 These initial steps are primarily used to identify any preexisting streams. 
#This information will be helpful later to distinguish between the preexisting streams and the new streams 
#created during the execution of the graph. This distinction is crucial because it allows the system to 
#determine which streams contain new data that needs to be processed (the program may also use old streams
#later we will see how to isolate new data from old data in old streams (step 8 and 9) to process new data). 

#Remember: streams here are like labeled containers, and identifying them helps keep track of where the 
#actual messages will be stored.

#Step 2: In this step, an initial scan of the existing streams 
#        in Redis is performed. The function r.scan(0, _type='stream') 
#        returns a cursor (curs) that is used to iterate through the 
#        existing streams. The variable start_streams is a list that
#        accumulates the names of the streams found during the scan.

curs, start_streams = r.scan(0, _type='stream') 
                                        
while curs != 0:
    curs, streams = r.scan(curs, _type='stream')# continue scan from the last cursor in previous batch
    start_streams += streams #update streams

#Step 3: Once the list of start_streams has been identified, the 
#        next step is to obtain the most recent identifier for each 
#        of these streams. This is done using r.xrevrange(stream, 
#        count=1) for each stream in start_streams. r.xrevrange is 
#        used to retrieve the reverse range (the most recent message)
#        from the stream. The result is a list of responses (replies),
#        and if this list is not empty (indicating that there are messages
#        in the stream), the identifier of the most recent message (which 
#        is the first element of the replies list) is taken and stored in 
#        the start_id dictionary with the stream's name as the key.

start_id = {}
for stream in start_streams:
    replies = r.xrevrange(stream, count=1)
    if replies:
        start_id[stream] = replies[0][0]

#Step 4: The json.dumps(graph) convert the graph in YAML format to json format, that is a more convenient format
#for web communication
print(f'Starting graph from {GRAPH}')
r.xadd('supervisor_ipstream', {'commands': 'startGraph','graph': json.dumps(graph)})

#Step 5 and 6
if DURATION:
    print(f'Waiting {DURATION} seconds')
    time.sleep(DURATION)
else:
    input('Hit ENTER to stop graph...')

# Stop the graph
print('Stopping graph')
r.xadd('supervisor_ipstream', {'commands': 'stopGraph'})

#Step 7: streams contains the name of all preexisting and new streams (basically we repeat step 2)
curs, stop_streams = r.scan(0, _type='stream')
while curs != 0:
    curs, streams = r.scan(curs, _type='stream')
    stop_streams += streams

#Step 8: It can be seen as:
#new_streams = []

#for stream in stop_streams:
#    if stream not in start_streams:
#        new_streams.append(stream)
new_streams = [
    stream for stream in stop_streams if stream not in start_streams
]

#Step 9
for stream in new_streams:
    start_id[stream] = 0 #Update the start_id dictionary with the new stream's names as the key

# Save streams data
all_data = {}
for stream in stop_streams:
    all_data[stream] = r.xrange(stream, min=start_id[stream])

# Some METADATA
date_str = datetime.now().strftime(r'%y%m%dT%H%M')
graph_name = os.path.splitext(os.path.basename(GRAPH))[0]
data_dir = os.path.join(test_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
save_path = os.path.join(data_dir, f'{date_str}_{graph_name}.pkl')
#Save stream data in file using pickle
with open(save_path, 'wb') as f:
    pickle.dump(all_data, f)
print(f'Saved streams: {sorted(list(all_data.keys()))}')

# Step 10:
# Remove saved data from Redis
# delete any streams created while the graph was running
i = 0
if new_streams:
    while max([r.xlen(stream) for stream in new_streams]):
        for stream in new_streams:
            r.delete(stream)
        i += 1
r.memory_purge()
print(f'Deleted streams: {new_streams}')

# constants
test_dir = os.getcwd()
data_dir = os.path.join(test_dir, 'data')
fig_dir = os.path.join(test_dir, 'figures')
data_file = save_path

# setup
os.makedirs(fig_dir, exist_ok=True)
with open(os.path.join(data_dir, data_file), 'rb') as f:
    graph_data = pickle.load(f)

# Load graph parameters
graphs = [
    json.loads(entry[b'graph']) for _, entry in graph_data[b'booter']
    if b'graph' in entry
]
graph = graphs[-1]

# Load info about the structure of each stream
with open('stream_NDS_spec_ol.yaml', 'r') as f:
    stream_spec = yaml.safe_load(f)

# Load and parse stream data
streams = [
    'BehaviorVelocity', 'Decoder', 'LFP', 'Position', 'Raw', 'SpikeEvents'
]
# Directly load and parse stream data without using YAML for stream specifications
decoded_streams = {}
for stream in streams:
    print(f'Processing {stream} stream')
    stream_data = graph_data.get(stream.encode(), [])  # Assumiendo que las claves son bytes
    out = []
    for entry_id, entry_data in tqdm(stream_data):
        entry_decoded = {}
        for key, val in entry_data.items():
            decoded_key = key.decode()
            # Decodifica el valor si es un string binario, y luego carga como JSON si el resultado es un string
            if isinstance(val, bytes):
                decoded_val = val.decode('utf-8')
                try:
                    # Intenta cargar el string decodificado como JSON
                    entry_decoded[decoded_key] = json.loads(decoded_val)
                except json.JSONDecodeError:
                    # Si no es un string JSON, simplemente usa el valor decodificado
                    entry_decoded[decoded_key] = decoded_val
            else:
                # Si no es un bytes, asume que ya está en un formato utilizable
                entry_decoded[decoded_key] = val
        out.append(entry_decoded)
    decoded_streams[stream] = pd.DataFrame(out)