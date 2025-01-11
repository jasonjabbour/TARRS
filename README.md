# TARRS

Reinforcement Learning for Recovering Time Synchronization in Robotic Networks

## Installation 

Devcontainer has all the requirements installed. 

Once you are in the Devcontainer run the following:

```bash
cd src/TARRS
pip install -e .
```

## Visualize Different Network Topologies
```bash
python3 safe_ptp/src/env/network_env.py
```

## Simulate and Visualize Attack on a Network

```bash
python3 safe_ptp/src/ptp/clock_sim_explorer.py
```

## Train an RL Agent
```bash
python3 safe_ptp/src/train.py
```

## Tips

To visualize through the Docker container, you may need to configure your `DISPLAY` environment variable to match the display port of your host machine. Use the following command:

```bash
export DISPLAY=<host_machine_display_port>
```
