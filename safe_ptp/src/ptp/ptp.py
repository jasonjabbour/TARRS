import simpy
import random
import tkinter as tk

from safe_ptp.src.ptp.viewer import Viewer
from safe_ptp.src.ptp.packet import Packet


class Clock:
    def __init__(self, env, name, error, drift, network, priority):
        self.env = env
        self.name = name
        self.error = error
        self.drift = drift
        self.time = 0
        self.leader = False
        self.offset = 0
        self.priority = priority
        self.network = network
        self.follow_up_pending = None
        self.delay_req_pending = None
        self.follow_up_time_stamp = None

    def run(self):
        while True:
            yield self.env.timeout(1)  # Clock updates every second
            self.time += 1 + self.drift  # Time adjustment including drift
            if self.leader and self.env.now % 10 == 0:  # Leader sends SYNC every 10 seconds
                self.broadcast_sync()                

    def receive_packet(self, packet):
        if packet.content['type'] == 'sync':
                # Store the SYNC reception time for later use in delay calculations
                self.sync_receive_time = self.time  # Record the time this node 'observed' the SYNC packet
                self.follow_up_pending = packet.id  # Set up expectation for corresponding follow-up
        elif packet.content['type'] == 'follow_up' and packet.content['sync_id'] == self.follow_up_pending:
            self.process_follow_up(packet)
            self.follow_up_pending = None  # Reset follow-up expectation after processing
        elif packet.content['type'] == 'delay_req':
            self.send_delay_response(packet)
        elif packet.content['type'] == 'delay_resp' and packet.content['req_id'] == self.delay_req_pending:
            self.process_delay_response(packet)
            self.delay_req_pending = None  # Reset delay request expectation

    def broadcast_sync(self):
        # Broadcasts a SYNC packet to all connected devices
        self.follow_up_time_stamp = self.time
        cpu_2_port_delay = random.expovariate(1.0)/10
        sync_packet = Packet(sender=self, receiver=None, content={'type': 'sync', 'time': self.follow_up_time_stamp + cpu_2_port_delay})
        self.network.send(sync_packet, self)
        self.env.process(self.send_follow_up(sync_packet))  # Schedule follow-up message

    def send_follow_up(self, sync_packet):
        yield self.env.timeout(0.2)  # Simulate processing delay for sending follow-up
        follow_up_packet = Packet(sender=self, receiver=None, content={'type': 'follow_up', 'sync_id': sync_packet.id, 'exact_time': self.follow_up_time_stamp})
        self.network.send(follow_up_packet, self)
        self.follow_up_time_stamp = None

    def process_follow_up(self, follow_up_packet):
        # Calculate the offset from the master clock
        exact_sync_send_time = follow_up_packet.content['exact_time']
        self.offset = self.sync_receive_time - exact_sync_send_time
        self.send_delay_request()

    def send_delay_request(self):
        if not self.leader and self.delay_req_pending is None:
            delay_req_packet = Packet(sender=self, receiver=None, content={'type': 'delay_req', 'time': self.time})
            self.delay_request_send_time = self.env.now  # Save the environment time of sending delay request
            self.network.send(delay_req_packet, self)
            self.delay_req_pending = delay_req_packet.id
            
    def send_delay_response(self, delay_req_packet):
        # Respond to a delay request
        delay_resp_packet = Packet(sender=self, receiver=delay_req_packet.sender, content={'type': 'delay_resp', 'req_id': delay_req_packet.id, 'time': self.time})
        self.network.send(delay_resp_packet, self)

    def process_delay_response(self, delay_resp_packet):
        delay_response_receive_time = self.env.now
        round_trip_delay = delay_response_receive_time - self.delay_request_send_time
        self.path_delay = round_trip_delay / 2

        # Calculate total adjustment
        self.time_adjustment = self.offset + (self.path_delay / 2)

        # Apply adjustment and log the times
        self.apply_time_adjustment()

    def apply_time_adjustment(self):
        if not self.leader:  # Only apply adjustment for non-leader clocks
            old_time = self.time
            self.time -= self.time_adjustment
            leader_time = self.network.find_leader().time
            print(f'Clock {self.name} synchronized. New Time: {self.time}, Leader Time: {leader_time}, Time Adjusted by: {self.time_adjustment}')
            print(f'{self.name} clock was at {old_time}, adjusted to {self.time}. Leader is at {leader_time}. Env time: {self.env.now}')
        else:
            leader_time = self.time
            print(f'Leader {self.name} time is: {leader_time}. Env time: {self.env.now}')



class NetworkSwitch:
    def __init__(self, env, name, gui):
        self.env = env
        self.name = name
        self.connections = []
        self.gui = gui
        self.network_delay = 1

    def find_leader(self):
        # Return the leader clock
        for device in self.connections:
            if device.leader:
                return device
        return None
    
    def start_clocks(self, clocks):
        for clock in clocks:
            self.env.process(clock.run())
    
    def connect_all(self, clocks):
        for clock in clocks:
            self.connect(clock)

    def connect(self, device):
        self.connections.append(device)

    def send(self, packet, origin):
        self.env.process(self.handle_packet(packet, origin))  # Start a new process for handling the packet

    def handle_packet(self, packet, origin):
        # TODO: Send network delay to visualization
        if self.gui:
            self.gui.animate_packet(origin, packet.content['type'], self.network_delay)
        yield self.env.timeout(self.network_delay)  # Wait for the delay
        for device in self.connections:
            if device != origin:
                device.receive_packet(packet)
                print(f'{self.env.now}: {self.name} forwarded {packet.content["type"]} packet from {packet.sender.name} to {device.name}.')




