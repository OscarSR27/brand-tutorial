# Hiwi Project - BRAND setup, LSL module, latency testing

Content of this repo is based on brand tutorial repo.
### Neural Data Simulator with BRAND

The Neural Data Simulator (NDS) is an open-source tool designed to simulate neuronal spikes from behavioral data. It incorporates cursor movement experiments to generate neuronal spike data. The objective is to integrate NDS data collection with BRAND, subsequently forwarding the collected data to Redis. This integration also encompasses latency measurements for data transmission from NDS to BRAND via Lab Streaming Layer (LSL), documenting the send/receive timestamps and samples for each stream in csv format within the DATA_test folder.

#### Setup and Configuration

Follow these instructions to configure and launch the data stream:

1. **NDS Installation:** Visit the [NDS Installation Page](https://agencyenterprise.github.io/neural-data-simulator/installation.html) and follow the instructions to install Neural Data Simulator on your system.

2. **BRAND Initialization:** Launch the BRAND application and initialize the supervisor to prepare for data collection and simulation.

3. **Launching NDS:** In a separate terminal start the NDS application by executing the `run_closed_loop` command. This initiates the simulation.

4. **Collect Data:** To record the NDS data, utilize one of the provided Jupyter notebooks or the Python script. Run either [NDS_calibration.ipynb](notebooks/NDS_calibration.ipynb) or [NDS_calibration.py](notebooks/NDS_calibration.py) according to your preference.

#### Files

- **Graph:** Utilize [sim_graph_NDS.yaml](notebooks/graphs/sim_graph_NDS.yaml) for structuring simulation data flow.
- **Stream Description:** The stream's configuration is defined in [stream_NDS_spec_ol.yaml](notebooks/stream_NDS_spec_ol.yaml).
- **Module Integration:** [my-modules folder](brand-modules/my-modules) contains files that implement the class responsible for reading data from NDS. To generate necessary `.bin` files within this directory, execute the `make` command from the root of the BRAND directory.

### Visualization of Spike Data using Open Ephys GUI

The Open Ephys GUI is a tool for visualizing neural data, including spike information, available at [Open Ephys GUI](https://open-ephys.org/gui). To effectively use this graphical interface for visualizing data from the BRAND framework, it's essential to bridge the data via Lab Streaming Layer (LSL) inlets. Therefore, the primary aim is to create an interface that connects data received via Redis in the BRAND framework with the Open Ephys GUI. By utilizing LSL for data transmission, this integration allows for the real-time visualization of neural spike data.

#### Files

Two main resources are provided to facilitate this task:

- **[OpenEphys_Steps.pdf](notebooks/OpenEphys_Steps.pdf)**: Offers instructions for the installation and utilization of the Open Ephys GUI. It serves as a foundational guide to ensure users can navigate and leverage the GUI effectively.
  
- **[redis_to_lsl.py](notebooks/redis_to_lsl.py)**: Implements a class designed for establishing a connection to Redis within the BRAND framework. This script ensures the reception of data, which it then formats appropriately (1x#channels) for transmission via LSL to the Open Ephys GUI. Upon execution, it also initiates an LSL inlet named "Spikes." This inlet is readily recognized by the Open Ephys GUI through its "LSL inlet" plugin.


### BRAND Tutorial

This repository contains a brief tutorial on how to get started using BRAND. The tutorial consists of the following steps:
1. Set up your system for BRAND by following the instructions in [00_setup.md](./notebooks/00_setup.md)
2. Run a calibration task and train a decoder using [01_calibration.ipynb](./notebooks/01_calibration.ipynb)
3. Run a closed-loop cursor control task with your trained decoder using [02_control.ipynb](./notebooks/02_control.ipynb)
4. If you have multiple machines on whicn you can run BRAND, try out the multi-machine version of this task with [03_multi_machine.ipynb](./notebooks/03_multi_machine.ipynb)

Once you have finished the above steps, you will have a working example of how to use BRAND to run an experiment.

Next, you may want to:   
- Learn how graphs are configured (see docs in [brand/README.md](https://github.com/brandbci/brand/blob/main/README.md) and examples in [notebooks/graphs](notebooks/graphs))
- Write finite-state machine and graphics nodes to fit the task of your choice (see [radialFSM.py](brand-modules/cursor-control/nodes/radialFSM/radialFSM.py) and [display_centerOut.py](brand-modules/cursor-control/nodes/display_centerOut/display_centerOut.py))
- Try out a new decoder architecture (see [wiener_filter.py](brand-modules/cursor-control/nodes/wiener_filter/wiener_filter.py))
- Acquire data from a neural recording device like a Blackrock Neurotech Neural Signal Processor (NSP) (see [cerebusAdapter.c](https://github.com/brandbci/brand-nsp/blob/main/nodes/cerebusAdapter/cerebusAdapter.c))

If you run into any issues, please document and submit them here on GitHub.
