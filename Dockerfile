FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Bogota \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential python3-dev libffi-dev libssl-dev \
      cron nano gcc supervisor tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --no-cache-dir && pip install poetry --no-cache-dir

COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-root

# app + cron + supervisor
COPY algo.py .
COPY helpers.py .
COPY log.py .
COPY SES.py .
COPY scheduler/run.sh /app/run.sh
COPY scheduler/crontab /etc/cron.d/weekly-job
COPY supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# permissions (no 'crontab' here!)
RUN chmod +x /app/run.sh \
 && chmod 0644 /etc/cron.d/weekly-job \
 && touch /var/log/cron.log

CMD ["/usr/bin/supervisord"]
