FROM node:22-bookworm-slim AS frontend-build

WORKDIR /app/webapp/frontend

COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci

COPY webapp/frontend/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . ./
COPY --from=frontend-build /app/webapp/frontend/dist /app/webapp/frontend/dist
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["./start.sh"]
