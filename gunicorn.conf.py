# Gunicorn configuration file

# Bind to host and port
bind = "0.0.0.0:10000"

# Number of workers (adjust based on CPU cores)
workers = 4  # Example: For a 2-core server, (2 * 2) + 1 = 5, but 4 is balanced

# Number of threads per worker (for I/O-bound tasks)
threads = 4

# Worker class (use 'gevent' for async, requires `pip install gevent`)
worker_class = "gevent"

# Preload the application to save memory
preload_app = True

# Timeout for worker processes (in seconds)
timeout = 30

# Keep-alive time for connections (in seconds)
keepalive = 2

# Logging configuration
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stdout
loglevel = "info"

# Maximum number of simultaneous clients per worker
worker_connections = 1000

# Restart workers to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50