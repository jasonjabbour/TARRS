import tkinter as tk


class Viewer:
    def __init__(self, master, clocks, network):
            self.master = master
            self.master.title("Network Simulation")
            self.canvas = tk.Canvas(master, width=600, height=400)
            self.canvas.pack()
            self.clock_items = {}
            self.connection_lines = {}

            # Draw clocks
            positions = [(100, 150), (500, 150)]  # Adjust based on the number of nodes
            for clock, pos in zip(clocks, positions):
                x, y = pos
                item = self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="blue", tags="clock")
                self.canvas.create_text(x, y, text=clock.name)
                self.clock_items[clock] = (item, x, y)

            # Draw connections
            for i, clock in enumerate(clocks):
                for j, other in enumerate(clocks):
                    if i != j:
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        line = self.canvas.create_line(x1, y1, x2, y2, tags="connection")
                        self.connection_lines[(clock, other)] = line

    def animate_packet(self, sender, receiver, packet_type):
        if (sender, receiver) in self.connection_lines:
            line = self.connection_lines[(sender, receiver)]
            x1, y1, x2, y2 = self.canvas.coords(line)

            # Create a packet representation
            packet = self.canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="red", tags="packet")
            packet_label = self.canvas.create_text(x1, y1 + 10, text=packet_type, fill="black", tags="packetLabel")

            steps = 20
            for i in range(1, steps + 1):
                # Calculate new positions for the packet and label
                new_x = x1 + (x2 - x1) * i / steps
                new_y = y1 + (y2 - y1) * i / steps

                # Move the packet and the label towards the receiver
                self.canvas.coords(packet, new_x - 5, new_y - 5, new_x + 5, new_y + 5)
                self.canvas.coords(packet_label, new_x, new_y + 10)
                
                self.master.update()
                self.canvas.after(50)  # Delay for animation effect

            # Remove the packet and label after the animation
            self.canvas.delete(packet)
            self.canvas.delete(packet_label)
  
    def update_clock(self, clock, message):
        item, x, y = self.clock_items[clock]
        self.canvas.itemconfig(item, fill="red" if clock.leader else "blue")
        self.canvas.create_text(x, y + 30, text=message, tags="info")
        self.master.update()

    def clear_info(self):
        # This now becomes a simple method that can be called directly without being a SimPy process
        self.canvas.delete("info")
