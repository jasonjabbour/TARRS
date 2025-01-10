import uuid

class Packet:
    def __init__(self, sender, receiver, content):
        self.sender = sender  # The clock sending the packet
        self.receiver = receiver  # Intended receiver, None if broadcast
        self.content = content  # Content of the packet (type and time info)
        self.id = uuid.uuid4()  # Unique identifier for the packet