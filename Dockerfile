FROM python:3.11-slim

# Install required system tools
RUN apt-get update && apt-get install -y \
    curl ca-certificates gnupg build-essential \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend and frontend
COPY backend ./backend
COPY frontend ./frontend

# --- Install backend dependencies ---
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# --- Build frontend with correct API URL ---
WORKDIR /app/frontend

# Create production environment file with correct API URL
RUN echo "VITE_API_URL=http://localhost:5000" > .env.production

RUN npm install && npm run build

# --- Install a static file server (serve) ---
RUN npm install -g serve


# Expose ports
EXPOSE 5000
EXPOSE 3000

# Use a start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]

# # # ---- FINAL COMMAND: run backend & static frontend ----
# # CMD ["sh", "-c", "serve -s /app/frontend/dist -l 3000 & python /app/backend/server.py"]

# # Start supervisor
# CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]