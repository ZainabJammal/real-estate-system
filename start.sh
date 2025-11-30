#!/bin/bash

# Start backend
cd /app/backend && python server.py &

# Start frontend  
cd /app/frontend && serve -s dist -l 3000 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?