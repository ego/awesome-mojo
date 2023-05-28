"""Mojo TCP Socket Client with PythonInterface"""
from PythonInterface import Python
from PythonObject import PythonObject

let time = Python.import_module('time')
let os = Python.import_module('os')
let socket = Python.import_module('socket')

let server = "10.8.95.167"
let port = 55555

print("Start TCP Client.")
let client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Connect to server", server, port)
client.connect((server, port))

let msg = PythonObject("Hello, Mojo🔥 TCP Client")
print("Sending data", msg)
client.send(msg.encode())

# wait a little bit
time.sleep(4)
print("Waiting ..")

while True:
    let data = client.recv(1024)
    if data:
        print("Message from Server", data)
    break

client.shutdown(1)
client.close()
print("Disconnect and close connection.")
