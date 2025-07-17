
# import requests
import subprocess
import json


# MCP sunucusunu başlat
process = subprocess.Popen(
    ["python", "weather_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

def send_request(request):
    print(">>", json.dumps(request))
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()
    response = process.stdout.readline()
    print("<<", response.strip())

# 1. initialize
send_request({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
})

# 2. notifications/initialized
send_request({
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
})

# 3. tools/list
send_request({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
})

# 4. tools/call - get_weather
send_request({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "get_weather",
        "arguments": {"city": "Berlin"}
    }
})

process.terminate()
