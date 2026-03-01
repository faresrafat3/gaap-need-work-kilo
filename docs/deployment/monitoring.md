# GAAP Monitoring & Observability

Comprehensive monitoring, logging, and alerting for GAAP deployments.

---

## Overview

GAAP provides built-in observability features:

- **Metrics** - Prometheus-compatible metrics
- **Logging** - Structured JSON logging
- **Health Checks** - Kubernetes-style probes
- **Tracing** - Request correlation IDs
- **Alerting** - Configurable alert rules

---

## Health Checks

### Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/api/health` | Detailed health status | Full system status |
| `/ready` | Kubernetes readiness | `{status: "ready"}` |
| `/live` | Kubernetes liveness | `{status: "alive"}` |
| `/status` | Public status page | System overview |

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": 1705315800.123,
  "uptime_seconds": 3600.45,
  "version": "1.0.0",
  "system": {
    "memory": {
      "total_mb": 16384,
      "available_mb": 8192,
      "percent_used": 50
    },
    "cpu_percent": 25.5
  },
  "database": {
    "status": "connected",
    "response_time_ms": 2.5
  },
  "providers": {
    "kimi": "available",
    "deepseek": "available"
  }
}
```

### Status Values

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| `healthy` | All systems operational | None |
| `degraded` | Some components have issues | Monitor closely |
| `critical` | Critical issues detected | Immediate action |

---

## Metrics

### Prometheus Endpoint

Metrics are exposed at `/metrics` in Prometheus format:

```
# HELP gaap_requests_total Total HTTP requests
# TYPE gaap_requests_total counter
gaap_requests_total{method="GET",endpoint="/api/health",status="200"} 100

# HELP gaap_request_duration_seconds Request duration
# TYPE gaap_request_duration_seconds histogram
gaap_request_duration_seconds_bucket{method="POST",endpoint="/api/chat",le="1.0"} 50

# HELP gaap_active_connections Active WebSocket connections
gaap_active_connections{channel="events"} 5

# HELP gaap_memory_usage_bytes Memory usage by tier
gaap_memory_usage_bytes{tier="working"} 1048576
```

### Metric Categories

#### Request Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gaap_requests_total` | Counter | Total HTTP requests |
| `gaap_request_duration_seconds` | Histogram | Request latency |
| `gaap_active_requests` | Gauge | In-flight requests |
| `gaap_requests_in_flight` | Gauge | Active connections |

#### Business Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gaap_chat_requests_total` | Counter | Chat API calls |
| `gaap_sessions_created_total` | Counter | Sessions created |
| `gaap_providers_requests_total` | Counter | Provider calls by name |

#### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gaap_memory_usage_bytes` | Gauge | Memory by tier |
| `gaap_database_connections` | Gauge | DB pool connections |
| `gaap_websocket_connections` | Gauge | Active WS connections |

### Custom Metrics

Record custom metrics in your code:

```python
from gaap.metrics import get_metrics

metrics = get_metrics()

# Counter
metrics.inc_counter("my_feature_usage", {"version": "v1"})

# Gauge
metrics.set_gauge("queue_size", len(queue))

# Histogram
metrics.record_histogram("processing_time", duration_seconds)
```

---

## Logging

### Log Format

Structured JSON logs with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "gaap.api.chat",
  "message": "Chat request completed",
  "correlation_id": "req-abc-123",
  "method": "POST",
  "path": "/api/chat",
  "status_code": 200,
  "duration_ms": 2450.5,
  "provider": "kimi",
  "extra": {
    "tokens_used": 150
  }
}
```

### Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | Detailed debugging info |
| `INFO` | Normal operations |
| `WARNING` | Unexpected but handled |
| `ERROR` | Errors requiring attention |
| `CRITICAL` | System failure |

### Configuration

```bash
# Environment variables
GAAP_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
GAAP_LOG_FORMAT=json         # json, text
GAAP_LOG_OUTPUT=stdout       # stdout, file, both

# File logging
GAAP_LOG_FILE=/var/log/gaap/app.log
GAAP_LOG_ROTATION=daily      # daily, size
GAAP_LOG_MAX_SIZE=100MB
GAAP_LOG_RETENTION=30        # days
```

### Correlation IDs

Track requests across services:

```python
from gaap.logging_config import get_correlation_id, set_correlation_id

# Get current correlation ID
correlation_id = get_correlation_id()

# Set correlation ID for new request
set_correlation_id("req-123")

# Automatically propagated in HTTP headers
# X-Correlation-ID: req-123
```

---

## Tracing

### Distributed Tracing

GAAP supports OpenTelemetry tracing:

```python
from gaap.tracing import get_tracer

tracer = get_tracer()

with tracer.start_as_current_span("process_request") as span:
    span.set_attribute("request.id", request_id)
    span.set_attribute("user.id", user_id)
    
    # Your code here
    result = process()
    
    span.set_attribute("result.status", "success")
```

### Trace Context

 propagated via:
- HTTP headers (`traceparent`, `tracestate`)
- Logs (correlation IDs)
- WebSocket messages

---

## Alerting

### Recommended Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Error Rate | `error_rate > 5%` | Critical |
| High Latency | `p95_latency > 5s` | Warning |
| Provider Down | `provider_health == 0` | Critical |
| Memory Usage | `memory_percent > 90%` | Warning |
| Disk Space | `disk_percent > 85%` | Warning |
| Rate Limiting | `rate_limited_requests > 10/min` | Warning |

### Prometheus Alert Rules

```yaml
# rules/gaap-alerts.yml
groups:
  - name: gaap
    rules:
      - alert: GAAPHighErrorRate
        expr: rate(gaap_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}%"

      - alert: GAAPHighLatency
        expr: histogram_quantile(0.95, rate(gaap_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"

      - alert: GAAPProviderDown
        expr: gaap_provider_health == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Provider {{ $labels.provider }} is down"
```

---

## Dashboards

### Grafana Dashboard

Import dashboard `gaap-dashboard.json`:

```json
{
  "dashboard": {
    "title": "GAAP Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(gaap_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(gaap_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Active Connections",
        "targets": [
          {
            "expr": "gaap_websocket_connections",
            "legendFormat": "{{channel}}"
          }
        ]
      }
    ]
  }
}
```

### Key Metrics to Monitor

| Metric | Target | Alert If |
|--------|--------|----------|
| Request rate | > 0 | = 0 for 2m |
| Error rate | < 1% | > 5% |
| p95 latency | < 2s | > 5s |
| Memory usage | < 70% | > 90% |
| DB connections | < 80% pool | > 95% pool |

---

## Log Aggregation

### Fluentd Configuration

```xml
<source>
  @type tail
  path /var/log/gaap/*.log
  pos_file /var/log/gaap/fluentd.pos
  tag gaap
  <parse>
    @type json
  </parse>
</source>

<filter gaap>
  @type grep
  <regexp>
    key level
    pattern ^(ERROR|CRITICAL)$
  </regexp>
</filter>

<match gaap>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name gaap-logs
</match>
```

### ELK Stack

```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
    volumes:
      - es-data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

  fluentd:
    image: fluent/fluentd:v1.16
    volumes:
      - ./fluent.conf:/fluentd/etc/fluent.conf
      - /var/log/gaap:/var/log/gaap
```

---

## Troubleshooting

### Common Issues

**No metrics appearing:**
```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Verify Prometheus config
cat /etc/prometheus/prometheus.yml
```

**Missing logs:**
```bash
# Check log level
echo $GAAP_LOG_LEVEL

# Check log output
ls -la /var/log/gaap/

# View recent logs
docker-compose logs -f --tail=100 gaap-backend
```

**High memory usage:**
```bash
# Check memory metrics
curl http://localhost:8000/api/system/metrics | jq '.memory'

# Check for memory leaks
ps aux | grep gaap
```

### Debug Mode

Enable debug logging:

```bash
export GAAP_LOG_LEVEL=DEBUG
export GAAP_LOG_FORMAT=text
```

---

## Best Practices

1. **Set SLOs** - Define service level objectives
2. **Alert on Symptoms** - Not causes
3. **Use Correlation IDs** - Track requests end-to-end
4. **Structured Logging** - Always use JSON
5. **Monitor Budget** - Track costs per session
6. **Provider Health** - Monitor all providers
7. **Test Alerts** - Regular fire drills

---

## Tools Integration

### Prometheus

```yaml
scrape_configs:
  - job_name: 'gaap'
    static_configs:
      - targets: ['gaap:8000']
    scrape_interval: 30s
```

### Datadog

```python
from datadog import statsd

statsd.increment('gaap.chat.request')
statsd.histogram('gaap.chat.latency', duration_ms)
```

### Sentry

```python
import sentry_sdk
from gaap.core.config import get_config

config = get_config()

sentry_sdk.init(
    dsn=config.sentry_dsn,
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
)
```

---

## See Also

- [Docker Deployment](./docker.md)
- [Kubernetes Deployment](./kubernetes.md)
- [API Documentation](../api/README.md)
