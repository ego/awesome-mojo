from PythonInterface import Python
from PythonObject import PythonObject

# Python fastapi
let fastapi = Python.import_module("fastapi")
let uvicorn = Python.import_module("uvicorn")

var app = fastapi.FastAPI()
var router = fastapi.APIRouter()

# tricky part
let py = Python()
let py_code = """lambda: 'Hello Mojo🔥!'"""
let py_obj = py.evaluate(py_code)
print(py_obj)

router.add_api_route("/mojo", py_obj)
app.include_router(router)

print("Start FastAPI WEB Server")
uvicorn.run(app)
print("Done")
