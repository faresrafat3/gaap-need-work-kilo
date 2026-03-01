# GAAP Monitoring Stack

Comprehensive monitoring and alerting solution for the GAAP (General-purpose AI Architecture Platform) project.

## Overview

This monitoring stack provides:

- **Metrics Collection** via Prometheus
- **Visualization** via Grafana
- **Alert Management** via AlertManager
- **Log Aggregation** via Loki & Promtail
- **System Monitoring** via Node Exporter & cAdvisor
- **Endpoint Monitoring** via Blackbox Exporter

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAAP Application                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  API     │  │  Metrics │  │  Logs    │  │  Status  │        │
│  │  (8000)  │  │  (/metrics)│  │  (JSON)  │  │  (/status)│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Prometheus  │  │   Loki       │  │ AlertManager │
│  (9090)      │  │  (3100)      │  │  (9093)      │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌──────────────┐
                   │   Grafana    │
                   │   (3000)     │
                   └──────────────┘
```

## Quick Start

### 1. Environment Setup

Create a `.env.monitoring` file:

```bash
# Grafana Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-secure-password
GRAFANA_ROOT_URL=http://localhost:3000

# AlertManager Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
PAGERDUTY_SERVICE_KEY=your-pagerduty-key
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-email-password
ALERTMANAGER_EXTERNAL_URL=http://localhost:9093

# GAAP Application
GAAP_LOG_DIR=./logs
```

### 2. Start the Monitoring Stack

```bash
# Start all services
docker-compose -f docker-compose.monitoring.yml up -d

# Start with Linux-specific exporters
docker-compose -f docker-compose.monitoring.yml --profile linux up -d

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f
```

### 3. Access the Services

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Grafana | http://localhost:3000 | admin / ${GRAFANA_ADMIN_PASSWORD} |
| Prometheus | http://localhost:9090 | - |
| AlertManager | http://localhost:9093 | - |
| Loki | http://localhost:3100 | - |

## Metrics

### Application Metrics (Prometheus)

The GAAP application exposes metrics at `/metrics`:

#### Request Metrics
- `gaap_api_requests_total` - Total HTTP requests by method, endpoint, status
- `gaap_api_request_duration_seconds` - Request latency histogram
- `gaap_api_active_requests` - Currently processing requests
- `gaap_api_request_size_bytes` - Request body sizes
- `gaap_api_response_size_bytes` - Response body sizes

#### Error Metrics
- `gaap_api_errors_total` - Error counts by layer, type, severity
- `gaap_api_error_rate` - Calculated error rate
- `gaap_api_exceptions_total` - Exception counts by type
- `gaap_api_http_errors_total` - HTTP error counts

#### Provider Metrics
- `gaap_providers_latency_seconds` - Provider API latency
- `gaap_providers_requests_total` - Provider request counts
- `gaap_providers_errors_total` - Provider error counts
- `gaap_providers_active_requests` - Active provider requests
- `gaap_providers_time_to_first_token_seconds` - Streaming TTFT
- `gaap_providers_tokens_per_second` - Token generation rate

#### Cost & Token Metrics
- `gaap_cost_tokens_total` - Token usage by type
- `gaap_cost_dollars_total` - Accumulated cost
- `gaap_cost_current_period_dollars` - Current period cost
- `gaap_cost_tokens_per_request` - Token distribution

#### System Metrics
- `gaap_system_memory_usage_bytes` - Memory breakdown
- `gaap_system_memory_usage_percent` - Memory percentage
- `gaap_system_cpu_usage_percent` - CPU by mode
- `gaap_system_disk_usage_bytes` - Disk usage
- `gaap_system_disk_usage_percent` - Disk percentage
- `gaap_system_open_file_descriptors` - FD count
- `gaap_system_network_io_bytes_total` - Network I/O

#### Business Metrics
- `gaap_business_chat_sessions_total` - Session counts
- `gaap_business_chat_sessions_active` - Active sessions
- `gaap_business_messages_total` - Message counts
- `gaap_business_users_active` - Active users

## Alerts

### Critical Alerts (Immediate Response)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `GAAPHighErrorRate` | Error rate > 5% for 2m | critical |
| `GAAPInstanceDown` | Instance not responding | critical |
| `GAAPDatabaseDown` | No DB connections | critical |
| `GAAPHighMemoryUsage` | Memory > 90% for 5m | critical |
| `GAAPLowDiskSpace` | Disk free < 10% for 5m | critical |
| `GAAPProviderDown` | Provider not responding | critical |

### Warning Alerts (Soon Response)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `GAAPHighLatency` | P95 latency > 2s for 3m | warning |
| `GAAPHighCPUUsage` | CPU > 80% for 10m | warning |
| `GAAPProviderHighErrorRate` | Provider errors > 10% | warning |
| `GAAPDailyCostThreshold` | Daily cost > $100 | warning |

### Info Alerts (Monitoring)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `GAAPHighTokenUsage` | Token rate > 1M/hour | info |

### Alert Routing

- **Critical**: PagerDuty + Slack #gaap-critical + Email
- **Warning**: Slack #gaap-warnings + Email
- **Info**: Slack #gaap-info only
- **Provider**: Slack #provider-alerts
- **Database**: Slack #database-alerts
- **Business**: Slack #business-metrics

## Dashboards

### GAAP Overview (`gaap-overview`)

System-wide metrics:
- Service status indicators
- Request rate & error rate
- P95 latency
- Active requests
- Daily cost
- Request rate by endpoint
- Error rate trends
- Latency percentiles
- Status code distribution
- Chat sessions & token usage

### GAAP Providers (`gaap-providers`)

Provider-specific metrics:
- Requests by provider
- P95 latency by provider
- Error rate by provider
- Active requests
- Request rate by provider/status
- Provider latency percentiles
- Model usage pie chart
- Token rate by model
- Cost by provider
- Cost breakdown table

### GAAP System (`gaap-system`)

Infrastructure metrics:
- Memory usage
- CPU usage
- Disk usage
- Open file descriptors
- CPU usage by mode
- Memory usage trends
- Memory breakdown
- Process memory
- Disk usage by mount
- Network I/O

## Logging

### Structured JSON Logging

GAAP uses structured JSON logging with:
- Correlation IDs for request tracing
- Request & Session IDs
- Severity levels
- Source location (file, line, function)
- Timestamp (RFC3339)

Example log entry:
```json
{
  "timestamp": "2026-02-28T12:34:56.789Z",
  "level": "INFO",
  "logger": "gaap.api.chat",
  "message": "Processing chat request",
  "correlation_id": "corr-abc123def456",
  "request_id": "req-789xyz",
  "session_id": "sess-012abc",
  "source": {
    "file": "chat.py",
    "line": 123,
    "function": "process_request",
    "module": "gaap.api.chat"
  }
}
```

### Log Queries (Loki)

```logql
# All error logs
{job="gaap"} |= "level":"ERROR"

# Logs for specific correlation ID
{job="gaap"} |= "corr-abc123def456"

# Chat API logs only
{job="gaap", logger="gaap.api.chat"}

# High latency requests
{job="gaap"} | json | duration_ms > 2000

# Failed requests
{job="gaap"} |= "status_code":"5"
```

## Tracing

OpenTelemetry integration provides distributed tracing:

```python
from gaap.tracing import trace_span, traced

# Context manager
with trace_span("db_query", {"table": "users"}):
    result = await db.execute()

# Decorator
@traced("process_request")
async def process_request(data):
    return await handle(data)

# Database queries
from gaap.tracing import trace_db_query
with trace_db_query("SELECT * FROM users", table="users"):
    cursor.execute(query)
```

Trace attributes include:
- HTTP method, path, status
- Database query details
- LLM provider/model info
- Token counts and costs
- Correlation IDs

## Status Page

The status page is available at `/status`:

```json
{
  "page": {
    "title": "GAAP System Status",
    "description": "Real-time status of the GAAP platform"
  },
  "status": {
    "indicator": "operational",
    "description": "All systems operational"
  },
  "components": [
    {
      "name": "Database",
      "status": "operational",
      "description": "SQLite database is accessible",
      "response_time_ms": 10.0
    }
  ],
  "metrics": {
    "total_requests": 15000,
    "success_rate": 99.5,
    "avg_latency_ms": 150,
    "active_sessions": 42
  }
}
```

## Configuration

### Prometheus Configuration

Edit `prometheus.yml` to:
- Add new scrape targets
- Adjust scrape intervals
- Configure remote storage
- Add recording rules

### AlertManager Configuration

Edit `alertmanager.yml` to:
- Configure notification channels
- Add routing rules
- Set inhibition rules
- Customize templates

### Grafana Configuration

Environment variables:
- `GF_SECURITY_ADMIN_USER` - Admin username
- `GF_SECURITY_ADMIN_PASSWORD` - Admin password
- `GF_SERVER_ROOT_URL` - External URL
- `GF_AUTH_ANONYMOUS_ENABLED` - Enable anonymous access

## Maintenance

### Backup

```bash
# Backup Prometheus data
docker run --rm -v gaap_prometheus-data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz -C /data .

# Backup Grafana data
docker run --rm -v gaap_grafana-data:/data -v $(pwd):/backup alpine tar czf /backup/grafana-backup.tar.gz -C /data .
```

### Cleanup

```bash
# Clean old Prometheus data (older than 30 days)
docker exec -it gaap-prometheus promtool tsdb analyze /prometheus

# Clean Loki data
docker exec -it gaap-loki logcli delete --selector='{job="gaap"}' --start="30d" --dry-run
```

### Updates

```bash
# Pull latest images
docker-compose -f docker-compose.monitoring.yml pull

# Restart services
docker-compose -f docker-compose.monitoring.yml up -d
```

## Troubleshooting

### No Metrics in Grafana

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify GAAP metrics endpoint: http://localhost:8000/metrics
3. Check network connectivity between containers

### Alerts Not Firing

1. Check Prometheus rules: http://localhost:9090/rules
2. Test alert expression in Prometheus UI
3. Verify AlertManager is receiving alerts: http://localhost:9093/#/alerts

### Missing Logs

1. Check Promtail status: `docker logs gaap-promtail`
2. Verify log file paths in `promtail-config.yml`
3. Check Loki ingestion: http://localhost:3100/ready

### High Memory Usage

1. Reduce Prometheus retention: `--storage.tsdb.retention.time=15d`
2. Limit Loki max streams
3. Increase scrape intervals
4. Filter unnecessary metrics

## Performance

### Resource Requirements

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| Prometheus | 0.5 cores | 1-2 GB | 50 GB |
| Grafana | 0.2 cores | 512 MB | 1 GB |
| Loki | 0.3 cores | 512 MB | 20 GB |
| AlertManager | 0.1 cores | 128 MB | 1 GB |

### Optimization Tips

- Use recording rules for complex queries
- Enable Prometheus WAL compression
- Configure Grafana query caching
- Use Loki label cardinality limits
- Filter high-cardinality metrics

## Security

### Default Security Measures

- Grafana: Strong password required, no anonymous access
- Prometheus: No authentication (use reverse proxy)
- AlertManager: Environment variables for secrets
- Loki: No external access by default

### Production Hardening

1. Enable HTTPS/TLS for all services
2. Use reverse proxy (nginx/traefik) with auth
3. Restrict network access
4. Rotate credentials regularly
5. Enable audit logging
6. Use network policies (Kubernetes)

## API Reference

### Health Check

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/health?detailed=true
```

### Metrics Export

```bash
curl http://localhost:8000/metrics
```

### Status Page

```bash
curl http://localhost:8000/status
```

## Support

- **Documentation**: https://docs.gaap.local/monitoring
- **Runbooks**: https://docs.gaap.local/runbooks
- **Issues**: https://github.com/your-org/gaap/issues
- **Slack**: #gaap-monitoring
