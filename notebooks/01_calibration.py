# Specify the path to your mouse
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
from brand.timing import timespecs_to_timestamps, timevals_to_timestamps
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

mouse_device = '/dev/input/event8'
DURATION = None  # seconds
GRAPH = 'sim_graph_ol.yaml'
REDIS_IP = '127.0.0.1'
REDIS_PORT = 6379
test_dir = os.getcwd()

with open(os.path.join(test_dir, 'graphs', GRAPH), 'r') as f:
    graph = yaml.safe_load(f)

# Edit graph
node_names = [node['name'] for node in graph['nodes']]
ma_idx = node_names.index('mouseAdapter')
graph['nodes'][ma_idx]['parameters']['mouse_device'] = mouse_device

#Step 1
r = redis.Redis(host=REDIS_IP, port=REDIS_PORT)

curs, start_streams = r.scan(0, _type='stream') 
                                        
while curs != 0:
    curs, streams = r.scan(curs, _type='stream')# continue scan from the last cursor in previous batch
    start_streams += streams #update streams

start_id = {}
for stream in start_streams:
    replies = r.xrevrange(stream, count=1)
    if replies:
        start_id[stream] = replies[0][0]

print(f'Starting graph from {GRAPH}')
r.xadd('supervisor_ipstream', {'commands': 'startGraph','graph': json.dumps(graph)})

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
with open('stream_spec_ol.yaml', 'r') as f:
    stream_spec = yaml.safe_load(f)

# Load and parse stream data
streams = [
    b'targetData', b'cursorData', b'mouse_vel', b'binned_spikes',
    b'control'
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

# binning
bs_df = pd.DataFrame(decoded_streams['binned_spikes'])
bs_df.set_index('sync', drop=False, inplace=True)
bs_df.columns = [col + '_bs' for col in bs_df.columns]

# autocue
ac_df = pd.DataFrame(decoded_streams['control'])
ac_df.set_index('sync', drop=False, inplace=True)
ac_df.columns = [col + '_ac' for col in ac_df.columns]

# join the dataframes
bin_df = cd_df.join(td_df).join(bs_df).join(ac_df)

bin_df.head()

# Train a decoder
SEQ_LEN = 15  # sequence length for the Wiener filter


def get_lagged_features(data, n_history: int = 4):
    """
    Lag the data along the time axis. Stack the lagged versions of the data
    along the feature axis.

    Parameters
    ----------
    data : array of shape (n_samples, n_features)
        Data to be lagged
    n_history : int, optional
        Number of bins of history to include in the lagged data, by default 4

    Returns
    -------
    lagged_features : array of shape (n_samples, n_history * n_features)
        Lagged version of the original data
    """
    assert n_history >= 0, 'n_history must be greater than or equal to 0'
    seq_len = n_history + 1
    lags = [None] * seq_len
    for i in range(seq_len):
        lags[i] = np.zeros_like(data)
        lags[i][i:, :] = data[:-i, :] if i > 0 else data
    lagged_features = np.hstack(lags)
    return lagged_features


neural_stream = 'binned_spikes'
kin_stream = 'control'
gain = 3

neural_data = np.vstack(bin_df['samples_bs'])
neural_data = get_lagged_features(neural_data, n_history=SEQ_LEN - 1)
kin_data = np.vstack(bin_df['samples_ac'])[:, :2] * gain

X_train, X_test, y_train, y_test = train_test_split(neural_data,
                                                    kin_data,
                                                    test_size=0.25,
                                                    shuffle=False)
# Fit the Ridge regression model
# Use k-fold cross-validation to select the weight of the L2 penalty
scorer = make_scorer(r2_score)
mdl = RidgeCV(alphas=np.logspace(2, 5, 4), cv=3, scoring=scorer)
mdl.fit(X_train, y_train)
y_test_pred = mdl.predict(X_test)

print(mdl.alpha_)
print(r2_score(y_test, y_test_pred))

# Save the trained model
file_desc = data_file.split('_')[0]

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f'{file_desc}_wf_seq_len_{SEQ_LEN}.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(mdl, f)

# Update the config of the closed-loop graph to load the saved model
for cl_graph_path in ['sim_graph_cl.yaml', 'sim_graph_cl_mm.yaml']:
    with open(os.path.join(test_dir, 'graphs', cl_graph_path), 'rb') as f:
        cl_graph = yaml.safe_load(f)

    node_names = [node['name'] for node in cl_graph['nodes']]
    wf_idx = node_names.index('wiener_filter')

    cl_graph['nodes'][wf_idx]['parameters']['model_path'] = os.path.abspath(
        model_path)
    cl_graph['nodes'][wf_idx]['parameters']['seq_len'] = SEQ_LEN

    # Save the edited config
    cl_graph_gen_path = list(os.path.splitext(cl_graph_path))
    cl_graph_gen_path.insert(-1, '_gen')
    cl_graph_gen_path = ''.join(cl_graph_gen_path)

    with open(os.path.join(test_dir, 'graphs', cl_graph_gen_path), 'w') as f:
        yaml.dump(cl_graph, f)

N = 10000
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
kin_pred = mdl.predict(neural_data)
axes[0].plot(kin_data[-N:, 0])
axes[0].plot(kin_pred[-N:, 0])
axes[1].plot(kin_data[-N:, 1])
axes[1].plot(kin_pred[-N:, 1])
plt.show()