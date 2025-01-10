import tkinter as tk
import math

class Viewer:
    def __init__(self, master, clocks, network):
        self.master = master
        self.master.title("Network Simulation")
        self.canvas = tk.Canvas(master, width=600, height=400)
        self.canvas.pack()
        self.clock_items = {}
        self.connection_lines = {}
        self.clock_labels = {}

        # Dynamically calculate positions
        positions = self.calculate_positions(len(clocks))

        # Draw clocks
        for clock, pos in zip(clocks, positions):
            x, y = pos
            item = self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="yellow", tags="clock")
            label = self.canvas.create_text(x, y + 40, text=f"{clock.name}\nSkew: 0", tags="clockLabel")
            self.clock_items[clock] = (item, x, y)
            self.clock_labels[clock] = label

        # Draw connections
        for i, clock in enumerate(clocks):
            for j, other in enumerate(clocks):
                if i != j:
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    line = self.canvas.create_line(x1, y1, x2, y2, tags="connection")
                    self.connection_lines[(clock, other)] = line

    def calculate_positions(self, num_clocks):
        radius = 150  # Radius of the circle on which clocks are placed
        center_x, center_y = 300, 200  # Center of the canvas
        angle_step = 360 / num_clocks
        positions = []
        for i in range(num_clocks):
            angle = math.radians(i * angle_step)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def animate_packet(self, sender, packet_type, delay):
        for receiver in self.connection_lines.keys():
            if receiver[0] == sender:
                line = self.connection_lines[receiver]
                x1, y1, x2, y2 = self.canvas.coords(line)

                # Create and animate packet as previously, but to all receivers
                packet = self.canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="red", tags="packet")
                packet_label = self.canvas.create_text(x1, y1 + 10, text=packet_type, fill="black", tags="packetLabel")

                steps = 20
                step_delay = int(delay * 1000 / steps)  # Adjust animation based on delay

                for i in range(1, steps + 1):
                    new_x = x1 + (x2 - x1) * i / steps
                    new_y = y1 + (y2 - y1) * i / steps
                    self.canvas.coords(packet, new_x - 5, new_y - 5, new_x + 5, new_y + 5)
                    self.canvas.coords(packet_label, new_x, new_y + 10)

                    self.master.update()
                    self.canvas.after(step_delay)

                self.canvas.delete(packet)
                self.canvas.delete(packet_label)
  
    def update_clock(self, clock, message, skew):
        item, x, y = self.clock_items[clock]
        # Update clock fill color based on leader status
        self.canvas.itemconfig(item, fill="red" if clock.leader else "blue")
        # Update clock label with the latest skew and message
        self.canvas.itemconfig(self.clock_labels[clock], text=f"{clock.name}\nSkew: {skew}\n{message}")
        self.master.update()

    def clear_info(self):
        # This method can now be used to clear any dynamic information displayed on the canvas
        self.canvas.delete("info")