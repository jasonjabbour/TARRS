import simpy
import random
import tkinter as tk
from safe_ptp.src.ptp.viewer import Viewer
from safe_ptp.src.ptp.bmca import BMCA
from safe_ptp.src.ptp.ptp import Clock, NetworkSwitch

clock_num = 2

def create_clocks(clock_num):
    clocks = []
    for i in range(clock_num):
        clock_name = f"Clock{i}"
        clock_error = random.uniform(0.1, 0.9)
        clock_drift = random.uniform(0.01, 0.9)
        clock_priority = random.randint(1, 255)  # Assign a random priority between 1 and 255
        clocks.append(Clock(env, clock_name, clock_error, clock_drift, network, clock_priority))
    return clocks
    
def run_simulation():
    global env 

    try:
        env.run(until=env.now + 1)  # Run simulation for one more step
    except StopIteration:
        return  # If the simulation is done, stop calling this function
    root.after(1, run_simulation)  # Schedule the next call

# Setup the Simulation Environment
env = simpy.Environment()
network = NetworkSwitch(env, 'Switch1', None)

# Create clocks
clocks = create_clocks(clock_num)

# Initialize the GUI
root = tk.Tk()
gui = Viewer(root, clocks, network)
# Link GUI to the network switch
network.gui = gui  

# Connect Clocks
network.connect_all(clocks)

# Start clocks
network.start_clocks(clocks)

# Elect Leader
leader_finder = BMCA(clocks, env)
leader_clock = leader_finder.bmca()

# Start the simulation
root.after(1, run_simulation)  
root.mainloop()