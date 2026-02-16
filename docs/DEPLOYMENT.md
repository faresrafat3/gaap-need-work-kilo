# GAAP Deployment Guide

This guide covers deploying GAAP in various environments.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Production Configuration](#production-configuration)
3. [Environment Variables](#environment-variables)
4. [Monitoring](#monitoring)
5. [Scaling](#scaling)
6. [Troubleshooting](#troubleshooting)

---

## Docker Deployment

### Quick Start

```bash
# Build image
docker build -t gaap .

# Run container
docker run -d \
  --name gaap \
  -p 8501:8501 \
  -p 8080:8080 \
  -e GROQ_API_KEY=gsk_... \
  -e CEREBRAS_API_KEY=csk_... \
  gaap
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  gaap:
    build: .
    ports:
      - "8501:8501"  # Web UI
      - "8080:8080"  # API
      - "9090:9090"  # Metrics
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - CEREBRAS_API_KEY=${CEREBRAS_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GAAP_ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit pandas plotly

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/data /app/logs

# Environment
ENV GAAP_ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Ports
EXPOSE 8501 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run
CMD ["python", "-m", "gaap.web.app"]
```

---

## Production Configuration

### Configuration File

```yaml
# config/production.yaml
system:
  name: GAAP-Production
  environment: production
  log_level: INFO
  metrics_enabled: true
  health_check_enabled: true

budget:
  monthly_limit: 5000.0
  daily_limit: 200.0
  per_task_limit: 10.0

execution:
  max_parallel_tasks: 10
  genetic_twin_enabled: true
  self_healing_enabled: true

security:
  sandbox_type: gvisor
  audit_enabled: true
  encryption_enabled: true

providers:
  - name: groq
    enabled: true
    priority: 85
    rate_limit_per_minute: 30
    
  - name: cerebras
    enabled: true
    priority: 95
    rate_limit_per_minute: 30
```

### Load Configuration

```python
from gaap.core import load_config

config = load_config("config/production.yaml")
engine = GAAPEngine(providers=..., budget=config.budget.daily_limit)
```

---

## Environment Variables

### Required Variables

```bash
# API Keys (at least one required)
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk_...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
```

### Optional Variables

```bash
# System
GAAP_ENVIRONMENT=production    # development, staging, production
GAAP_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
GAAP_LOG_FORMAT=json           # json, text

# Budget
GAAP_BUDGET_MONTHLY=5000.0
GAAP_BUDGET_DAILY=200.0

# Execution
GAAP_MAX_PARALLEL_TASKS=10
GAAP_TIMEOUT=120

# Security
GAAP_SANDBOX_TYPE=gvisor
GAAP_AUDIT_ENABLED=true

# Monitoring
GAAP_METRICS_ENABLED=true
GAAP_METRICS_PORT=9090
GAAP_TRACING_ENABLED=true
```

### Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gaap-secrets
type: Opaque
stringData:
  GROQ_API_KEY: gsk_...
  CEREBRAS_API_KEY: csk_...
  GEMINI_API_KEY: ...
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: gaap-config
data:
  GAAP_ENVIRONMENT: production
  GAAP_LOG_LEVEL: INFO
  GAAP_BUDGET_DAILY: "200"
```

---

## Monitoring

### Prometheus Metrics

GAAP exposes Prometheus metrics on port 9090:

```
# HELP gaap_requests_total Total requests processed
# TYPE gaap_requests_total counter
gaap_requests_total{status="success"} 1234
gaap_requests_total{status="failed"} 45

# HELP gaap_request_duration_seconds Request duration
# TYPE gaap_request_duration_seconds histogram
gaap_request_duration_seconds_bucket{le="0.1"} 100
gaap_request_duration_seconds_bucket{le="1.0"} 500

# HELP gaap_tokens_used_total Total tokens used
# TYPE gaap_tokens_used_total counter
gaap_tokens_used_total{provider="groq"} 1000000

# HELP gaap_cost_dollars_total Total cost in dollars
# TYPE gaap_cost_dollars_total counter
gaap_cost_dollars_total 15.23
```

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gaap'
    static_configs:
      - targets: ['gaap:9090']
```

### Grafana Dashboard

Import dashboard with panels for:
- Request rate and success rate
- Latency percentiles (p50, p95, p99)
- Token usage by provider
- Cost tracking
- Memory usage

### Health Check Endpoint

```
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "providers": {
    "groq": "healthy",
    "cerebras": "healthy"
  },
  "memory_mb": 512
}
```

---

## Scaling

### Horizontal Scaling

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gaap
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gaap
  template:
    metadata:
      labels:
        app: gaap
    spec:
      containers:
      - name: gaap
        image: gaap:latest
        ports:
        - containerPort: 8501
        - containerPort: 8080
        - containerPort: 9090
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        envFrom:
        - secretRef:
            name: gaap-secrets
        - configMapRef:
            name: gaap-config
```

### Load Balancer

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gaap
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
    name: api
  - port: 8501
    targetPort: 8501
    name: web
  selector:
    app: gaap
```

### Auto-scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gaap-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gaap
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Cloud Deployment

### AWS

```bash
# ECR
aws ecr create-repository --repository-name gaap

# Build and push
docker build -t gaap .
docker tag gaap:latest <account>.dkr.ecr.<region>.amazonaws.com/gaap:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/gaap:latest

# ECS task definition
{
  "family": "gaap",
  "containerDefinitions": [{
    "name": "gaap",
    "image": "<account>.dkr.ecr.<region>.amazonaws.com/gaap:latest",
    "essential": true,
    "portMappings": [
      {"containerPort": 8501, "protocol": "tcp"},
      {"containerPort": 8080, "protocol": "tcp"}
    ],
    "environment": [
      {"name": "GAAP_ENVIRONMENT", "value": "production"}
    ],
    "secrets": [
      {"name": "GROQ_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
    ]
  }]
}
```

### Google Cloud

```bash
# GCR
gcloud builds submit --tag gcr.io/<project>/gaap

# Cloud Run
gcloud run deploy gaap \
  --image gcr.io/<project>/gaap \
  --platform managed \
  --region us-central1 \
  --set-env-vars GAAP_ENVIRONMENT=production \
  --set-secrets GROQ_API_KEY=projects/.../secrets/groq-key:latest
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs gaap

# Check health
docker inspect gaap | jq '.[0].State.Health'

# Common causes:
# - Missing environment variables
# - Invalid API keys
# - Resource limits exceeded
```

#### High Memory Usage

```bash
# Check memory
docker stats gaap

# Reduce parallelism
GAAP_MAX_PARALLEL_TASKS=3

# Enable memory guard
GAAP_MEMORY_GUARD_ENABLED=true
GAAP_MEMORY_LIMIT_MB=2048
```

#### Provider Failures

```bash
# Check provider status
gaap providers test --all

# Check rate limits
gaap status

# Solutions:
# - Add more API keys
# - Enable fallback providers
# - Reduce request rate
```

### Debugging in Production

```bash
# Enable debug logging
GAAP_LOG_LEVEL=DEBUG

# View logs
docker logs -f gaap

# Execute into container
docker exec -it gaap /bin/bash

# Run diagnostics
gaap doctor
```

---

## Security Checklist

- [ ] All API keys stored as secrets (not in code)
- [ ] HTTPS enabled for all endpoints
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Regular security updates applied
- [ ] Resource limits set
- [ ] Health checks configured
- [ ] Backup strategy in place

---

## Maintenance

### Backup

```bash
# Backup data directory
tar -czf gaap-backup-$(date +%Y%m%d).tar.gz data/ logs/

# Backup to S3
aws s3 cp gaap-backup.tar.gz s3://bucket/backups/
```

### Updates

```bash
# Pull latest image
docker pull gaap:latest

# Recreate container
docker-compose up -d --force-recreate

# Verify
gaap doctor
```

### Log Rotation

```yaml
# logrotate config
/app/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

## Next Steps

- [Security Guide](SECURITY.md) - Security best practices
- [Architecture Guide](ARCHITECTURE.md) - System architecture
- [API Reference](API_REFERENCE.md) - API documentation