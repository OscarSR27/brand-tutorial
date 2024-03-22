import json
import os
import pickle
import time
from datetime import datetime

import redis
import yaml
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

mouse_device = '/dev/input/event8'

DURATION = None  # seconds
GRAPH = 'sim_graph_cl_gen.yaml'
REDIS_IP = '127.0.0.1'
REDIS_PORT = 6379
test_dir = os.getcwd()

with open(os.path.join(test_dir, 'graphs', GRAPH), 'r') as f:
    graph = yaml.safe_load(f)

# Edit graph
node_names = [node['name'] for node in graph['nodes']]
ma_idx = node_names.index('mouseAdapter')
graph['nodes'][ma_idx]['parameters']['mouse_device'] = mouse_device

r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)

curs, start_streams = r.scan(0, _type='stream')
while curs != 0:
    curs, streams = r.scan(curs, _type='stream')
    start_streams += streams

# get the most recent ID from each stream
start_id = {}
for stream in start_streams:
    replies = r.xrevrange(stream, count=1)
    if replies:
        start_id[stream] = replies[0][0]

print(f'Starting graph from {GRAPH}')
r.xadd('supervisor_ipstream', {
    'commands': 'startGraph',
    'graph': json.dumps(graph)
})

if DURATION:
    print(f'Waiting {DURATION} seconds')
    time.sleep(DURATION)
else:
    input('Hit ENTER to stop graph...')

# Stop the graph
print('Stopping graph')
r.xadd('supervisor_ipstream', {'commands': 'stopGraph'})

curs, stop_streams = r.scan(0, _type='stream')
while curs != 0:
    curs, streams = r.scan(curs, _type='stream')
    stop_streams += streams

new_streams = [
    stream for stream in stop_streams if stream not in start_streams
]

for stream in new_streams:
    start_id[stream] = 0

# Save streams
all_data = {}
for stream in stop_streams:
    all_data[stream] = r.xrange(stream, min=start_id[stream])

date_str = datetime.now().strftime(r'%y%m%dT%H%M')
graph_name = os.path.splitext(os.path.basename(GRAPH))[0]
data_dir = os.path.join(test_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
save_path = os.path.join(data_dir, f'{date_str}_{graph_name}.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(all_data, f)
print(f'Saved streams: {sorted(list(all_data.keys()))}')

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
with open('stream_spec_cl.yaml', 'r') as f:
    stream_spec = yaml.safe_load(f)

# Load and parse stream data
streams = [
    b'targetData', b'cursorData', b'mouse_vel', b'binned_spikes',
    b'wiener_filter'
]
decoded_streams = {}
for stream in streams:
    print(f'Processing {stream.decode()} stream')
    stream_data = graph_data[stream]
    out = [None] * len(stream_data)
    spec = stream_spec[stream.decode()]
    for i, (entry_id, entry_data) in tqdm(enumerate(stream_data)):
        entry_dec = {}
        for key, val in entry_data.items():
            if key.decode() in spec:
                dtype = spec[key.decode()]
                if dtype == 'str':
                    entry_dec[key.decode()] = val.decode()
                elif dtype == 'sync':
                    entry_dec[key.decode()] = json.loads(val)['count']
                elif dtype == 'timeval':
                    entry_dec[key.decode()] = timevals_to_timestamps(val)
                elif dtype == 'timespec':
                    entry_dec[key.decode()] = timespecs_to_timestamps(val)
                else:
                    dat = np.frombuffer(val, dtype=dtype)
                    entry_dec[key.decode()] = dat[0] if dat.size == 1 else dat
        out[i] = entry_dec
    decoded_streams[stream.decode()] = out

# Load data at the binned spikes sample rate
# FSM
cd_df = pd.DataFrame(decoded_streams['cursorData'])
cd_df.set_index('sync', drop=False, inplace=True)
cd_df.columns = [col + '_cd' for col in cd_df.columns]

td_df = pd.DataFrame(decoded_streams['targetData'])
td_df.set_index('sync', drop=False, inplace=True)
td_df['angle'] = np.degrees(np.arctan2(td_df['Y'], td_df['X']))
td_df.columns = [col + '_td' for col in td_df.columns]

# decoding
wf_df = pd.DataFrame(decoded_streams['wiener_filter'])
wf_df.set_index('sync', drop=False, inplace=True)
wf_df.columns = [col + '_wf' for col in wf_df.columns]

# binning
bs_df = pd.DataFrame(decoded_streams['binned_spikes'])
bs_df.set_index('sync', drop=False, inplace=True)
bs_df.columns = [col + '_bs' for col in bs_df.columns]

# join the dataframes
bin_df = cd_df.join(td_df).join(wf_df).join(bs_df)

# plot kinematics
plt.plot(bin_df['X_cd'], bin_df['Y_cd'])
plt.axis('equal')
plt.ylabel('Y position')
plt.xlabel('X position')
plt.show()