# GAAP Deployment Guide

Deploy GAAP to production environments.

---

## Deployment Options

| Method | Best For | Complexity | Scalability |
|--------|----------|------------|-------------|
| [Docker Compose](./docker.md) | Single server, development | Low | Limited |
| [Kubernetes](./kubernetes.md) | Production, high availability | Medium | High |
| [Cloud](./cloud.md) | Managed services | Low-Medium | Very High |

### Quick Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│                     Choose Your Path                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Single server or testing?                                   │
│     └──► [Docker Compose](./docker.md)                      │
│                                                              │
│  Production with team?                                       │
│     └──► [Kubernetes](./kubernetes.md)                      │
│                                                              │
│  Want managed infrastructure?                                │
│     └──► [Cloud Deployment](./cloud.md)                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Pre-Deployment Checklist

### Security

- [ ] Changed default secrets
- [ ] Configured HTTPS/TLS
- [ ] Set up authentication
- [ ] Reviewed CORS settings
- [ ] Enabled audit logging
- [ ] Configured rate limiting
- [ ] Run security audit: `make security-audit`

### Configuration

- [ ] Set `GAAP_ENVIRONMENT=production`
- [ ] Configured database (PostgreSQL recommended)
- [ ] Set up Redis (for distributed rate limiting)
- [ ] Configured LLM API keys
- [ ] Set budget limits
- [ ] Reviewed feature flags

### Monitoring

- [ ] Enabled Prometheus metrics
- [ ] Set up log aggregation
- [ ] Configured health checks
- [ ] Set up alerting
- [ ] Tested failover procedures

---

## Environment Configuration

### Required for Production

```bash
# Environment
GAAP_ENVIRONMENT=production
GAAP_LOG_LEVEL=INFO
GAAP_LOG_FORMAT=json

# Security
GAAP_JWT_SECRET=<strong-random-secret>
GAAP_CAPABILITY_SECRET=<strong-random-secret>
GAAP_ENCRYPTION_KEY=<32-byte-hex>

# Database (PostgreSQL recommended)
DATABASE_URL=postgresql://user:pass@localhost/gaap

# LLM Provider (at least one)
GAAP_KILO_API_KEY=your_key_here
# OR
GAAP_OPENAI_API_KEY=your_key_here

# Redis (for distributed rate limiting)
GAAP_REDIS_URL=redis://localhost:6379/0
```

### Optional but Recommended

```bash
# Budget limits
GAAP_BUDGET_MONTHLY_LIMIT=5000
GAAP_BUDGET_DAILY_LIMIT=200

# Monitoring
GAAP_METRICS_ENABLED=true
GAAP_SENTRY_DSN=your_sentry_dsn

# Features
GAAP_ENABLE_SELF_HEALING=true
GAAP_ENABLE_EXTERNAL_RESEARCH=true
```

---

## Documentation Sections

- [Docker Deployment](./docker.md) - Docker Compose setup
- [Kubernetes](./kubernetes.md) - K8s manifests and Helm
- [Cloud](./cloud.md) - AWS, GCP, Azure deployment
- [Monitoring](./monitoring.md) - Observability setup

---

## Production Considerations

### Performance

- Use PostgreSQL instead of SQLite
- Enable connection pooling
- Configure CDN for static assets
- Set up caching layer

### Security

- Run containers as non-root
- Use secrets management (Vault, AWS Secrets Manager)
- Enable network policies
- Regular security updates

### Scaling

- Horizontal scaling with load balancer
- Database read replicas
- Redis cluster for session store
- Auto-scaling policies

---

## Support

Need help with deployment?

- 📖 [Troubleshooting](../../TROUBLESHOOTING.md)
- 💬 [GitHub Discussions](https://github.com/gaap-system/gaap/discussions)
- 📧 Email: ops@gaap.io
