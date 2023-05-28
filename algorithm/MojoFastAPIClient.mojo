from PythonInterface import Python
let http_client = Python.import_module("http.client")

let conn = http_client.HTTPConnection("localhost", 8000)
conn.request("GET", "/mojo")
let response = conn.getresponse()
print(response.status, response.reason, response.read())
