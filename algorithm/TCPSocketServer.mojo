"""Mojo TCP Socket Server with PythonInterface"""
from PythonInterface import Python

# Python Socket
let socket = Python.import_module('socket')

let host = socket.gethostbyname(socket.gethostname()).to_string()
print(host)
py_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

py_socket.bind(("10.8.95.167", 55555))
print("Socket bind to", "10.8.95.167", 55555)

py_socket.listen()
print("Listen for client connection ..")

var conn_addr = py_socket.accept()
var conn = conn_addr[0]
let addr = conn_addr[1]
print("Connected by client", addr)

while True:
    let data = conn.recv(1024)
    if data:
        print("Message from client", data)
    else:
        break
    # ping-pong
    conn.sendall(data)

conn.close()
print("Connection close.")

py_socket.close()
print("Socket close.")
