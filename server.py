# This is TEST SERVER for learning websockets
# NOT REQUIRED FOR FINALL PRODUCT

import socket, struct

address = ("127.0.0.1", 4242)
buf = bytearray(8 * 6)
data = [1, 2, 3, 40, 50, 0]

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    while True:
        struct.pack_into('dddddd', buf, 0, *data)
        sock.sendto(buf, address)