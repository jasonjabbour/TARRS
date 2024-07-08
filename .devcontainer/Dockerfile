# Use Ubuntu 22.04 LTS as the base image
FROM ubuntu:22.04

# Set noninteractive installation to avoid timezone prompt
ARG DEBIAN_FRONTEND=noninteractive

# Install Python, pip and other necessary utilities
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    x11-apps \
    sudo \
    python3-tk

# Upgrade pip to its latest version
RUN python3 -m pip install --upgrade pip

# Install required Python modules
RUN pip install matplotlib stable-baselines3 gymnasium networkx simpy torch_geometric

RUN pip install tensorboard

# IF USING TFAGENTS: 
# # Install TensorFlow, ensure you get a version that matches your Python setup
# RUN pip install tensorflow==2.15.0 tensorflow-probability==0.19.0

# # Install TF Agents stable version
# RUN pip install tf-agents

# Define the default working directory where your source code is located
WORKDIR /tmp/safe_ptp/src/safe_ptp

# Define default command to keep the container running
CMD ["/bin/bash"]

