import os

# Bind to Render's dynamic port or fallback to 10000
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 1  # Reduced from 4 to minimize memory usage
threads = 2  # Reduced from 4 to lower overhead
worker_class = "sync"  # Keep async worker for concurrency
preload_app = True  # Load app once to save memory
timeout = 30  # Keep default timeout
keepalive = 2  # Keep default keep-alive
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stdout
loglevel = "info"  # Keep info-level logging
worker_connections = 100  # Reduced from 1000 to limit memory
max_requests = 500  # Restart workers sooner to prevent leaks
max_requests_jitter = 50  # Keep jitter for randomization