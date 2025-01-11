# safe_ptp

## Installation 

Devcontainer has all the requirements installed. 

Once you are in the Devcontainer run the following:

```bash
cd src/safe_ptp
pip install -e .
```

## Simulate Attack on a Network

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



## Trouble Shooting
Some packages that might be helpful. 

```bash
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglib2.0-0
sudo apt-get install swig

pip install opencv-python
pip install gymnasium[classic-control]
pip install gymnasium[box2d]
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install shimmy
```