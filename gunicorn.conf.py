# Gunicorn configuration file

# Bind to host and port (unchanged, compatible with Render)
bind = "0.0.0.0:10000"

# Number of workers: Use 1 worker to minimize memory usage on Render’s low-memory plans
workers = 1

# Number of threads: Use 2 threads to balance concurrency and memory usage
threads = 2

# Worker class: Keep 'gevent' for I/O-bound tasks (CSV processing, visualization)
worker_class = "gevent"

# Preload the application to save memory
preload_app = True

# Timeout: Increase to 300 seconds to handle large CSV processing and visualization
timeout = 300

# Keep-alive: Increase to 5 seconds to reduce connection overhead for low-traffic apps
keepalive = 5

# Logging configuration (unchanged, suitable for Render)
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stdout
loglevel = "info"

# Worker connections: Reduce to 100 for moderate concurrency
worker_connections = 100

# Restart workers to prevent memory leaks (slightly lower to ensure frequent refresh)
max_requests = 500
max_requests_jitter = 50