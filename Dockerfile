FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Required system build packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    cron \
    nano \
    gcc \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip --no-cache-dir && pip install poetry --no-cache-dir

# Copy dependency files early for caching
COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-root

# Copy app files
COPY algo.py .
COPY scheduler/run.sh /app/run.sh
COPY scheduler/crontab /etc/cron.d/weekly-job
COPY supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# File permissions
RUN chmod +x /app/run.sh && chmod 0644 /etc/cron.d/weekly-job
RUN crontab /etc/cron.d/weekly-job

# Create log file
RUN touch /var/log/cron.log

# Supervisor will manage processes
CMD ["/usr/bin/supervisord"]
