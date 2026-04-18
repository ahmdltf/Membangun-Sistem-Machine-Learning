from flask import Flask
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# 10 metrics (ADVANCE)
request_count = Counter('request_count', 'Total Request')
success_count = Counter('success_count', 'Success Request')
error_count = Counter('error_count', 'Error Request')
latency = Counter('latency', 'Latency')
cpu_usage = Counter('cpu_usage', 'CPU Usage')
memory_usage = Counter('memory_usage', 'Memory Usage')
disk_usage = Counter('disk_usage', 'Disk Usage')
network_usage = Counter('network_usage', 'Network Usage')
prediction_count = Counter('prediction_count', 'Prediction Count')
model_loaded = Counter('model_loaded', 'Model Loaded')

@app.route("/metrics")
def metrics():
    return generate_latest()

@app.route("/")
def home():
    request_count.inc()
    success_count.inc()
    prediction_count.inc()
    return "OK"

if __name__ == "__main__":
    app.run(port=8000)
