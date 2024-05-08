import simpy
import random
import tkinter as tk
from viewer import Viewer
from bmca import BMCA
from ptp import Clock, NetworkSwitch


def run_simulation():
    try:
        env.run(until=env.now + 1)  # Run simulation for one more step
    except StopIteration:
        return  # If the simulation is done, stop calling this function
    root.after(100, run_simulation)  # Schedule the next call

# Setup the GUI and Simulation Environment
root = tk.Tk()
env = simpy.Environment()
network = NetworkSwitch(env, 'Switch1', None)

clock_num = 2
clocks = []
clock_drift_lst = [0,.345, .566, .567]

for i in range(clock_num):
    clock_name = f'Clock{i}'
    clock_accuracy = random.uniform(0.1, 0.9)
    clock_drift = clock_drift_lst[i]
    clocks.append(Clock(env, clock_name, clock_accuracy, clock_drift, network))

gui = Viewer(root, clocks, network)
network.gui = gui  # Now the GUI is correctly linked to the network switch

# Connect clocks and elect leader
for clock in clocks:
    network.connect(clock)

leader_finder = BMCA(clocks, env)
leader_clock = leader_finder.bmca()

# Start clock processes
for clock in clocks:
    env.process(clock.run())

root.after(100, run_simulation)  # Start the simulation
root.mainloop()