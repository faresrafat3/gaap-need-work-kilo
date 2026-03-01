# Kubernetes Deployment

Deploy GAAP on Kubernetes for production scalability.

---

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3 (optional but recommended)

## Quick Start

```bash
# Create namespace
kubectl create namespace gaap

# Apply manifests
kubectl apply -f k8s/ -n gaap

# Check status
kubectl get pods -n gaap
kubectl get svc -n gaap
```

---

## Kubernetes Manifests

### 1. Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: gaap
  labels:
    app: gaap
    environment: production
```

### 2. ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gaap-config
  namespace: gaap
data:
  GAAP_ENVIRONMENT: "production"
  GAAP_LOG_LEVEL: "INFO"
  GAAP_LOG_FORMAT: "json"
  GAAP_METRICS_ENABLED: "true"
  CORS_ORIGINS: "https://gaap.example.com"
```

### 3. Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: gaap-secrets
  namespace: gaap
type: Opaque
stringData:
  GAAP_JWT_SECRET: "your-jwt-secret"
  GAAP_CAPABILITY_SECRET: "your-capability-secret"
  GAAP_KILO_API_KEY: "your-api-key"
  DATABASE_URL: "postgresql://user:pass@postgres:5432/gaap"
```

Create from command line:

```bash
kubectl create secret generic gaap-secrets \
  --from-literal=GAAP_JWT_SECRET=$(openssl rand -hex 32) \
  --from-literal=GAAP_KILO_API_KEY=your_key \
  -n gaap
```

### 4. Backend Deployment

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gaap-backend
  namespace: gaap
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gaap-backend
  template:
    metadata:
      labels:
        app: gaap-backend
    spec:
      containers:
      - name: backend
        image: gaap/backend:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: gaap-config
        - secretRef:
            name: gaap-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
```

### 5. Backend Service

```yaml
# k8s/backend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gaap-backend
  namespace: gaap
spec:
  selector:
    app: gaap-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### 6. Frontend Deployment

```yaml
# k8s/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gaap-frontend
  namespace: gaap
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gaap-frontend
  template:
    metadata:
      labels:
        app: gaap-frontend
    spec:
      containers:
      - name: frontend
        image: gaap/frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: PYTHON_API_URL
          value: "http://gaap-backend:8000"
        - name: NODE_ENV
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 7. Frontend Service

```yaml
# k8s/frontend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gaap-frontend
  namespace: gaap
spec:
  selector:
    app: gaap-frontend
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
```

### 8. Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gaap-ingress
  namespace: gaap
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt"
spec:
  tls:
  - hosts:
    - gaap.example.com
    secretName: gaap-tls
  rules:
  - host: gaap.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: gaap-backend
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: gaap-frontend
            port:
              number: 3000
```

### 9. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gaap-backend-hpa
  namespace: gaap
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gaap-backend
  minReplicas: 3
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

## Helm Chart

### Chart Structure

```
helm/gaap/
├── Chart.yaml
├── values.yaml
├── values-production.yaml
└── templates/
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    └── secrets.yaml
```

### values.yaml

```yaml
# helm/gaap/values.yaml
replicaCount: 2

image:
  backend:
    repository: gaap/backend
    tag: latest
    pullPolicy: IfNotPresent
  frontend:
    repository: gaap/frontend
    tag: latest
    pullPolicy: IfNotPresent

service:
  type: ClusterIP
  backendPort: 8000
  frontendPort: 3000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: gaap.example.com
      paths:
        - path: /api
          service: backend
        - path: /
          service: frontend
  tls:
    - secretName: gaap-tls
      hosts:
        - gaap.example.com

resources:
  backend:
    requests:
      memory: 512Mi
      cpu: 500m
    limits:
      memory: 1Gi
      cpu: 1000m
  frontend:
    requests:
      memory: 256Mi
      cpu: 250m
    limits:
      memory: 512Mi
      cpu: 500m

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  GAAP_ENVIRONMENT: production
  GAAP_LOG_LEVEL: INFO
```

### Install with Helm

```bash
# Add repo (if published)
helm repo add gaap https://charts.gaap.io

# Install
cd helm/gaap
helm install gaap . -n gaap --create-namespace

# Upgrade
helm upgrade gaap . -n gaap

# With custom values
helm install gaap . -n gaap -f values-production.yaml
```

---

## PostgreSQL Deployment

### Using Helm

```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami

# Install PostgreSQL
helm install postgres bitnami/postgresql \
  --namespace gaap \
  --set auth.username=gaap \
  --set auth.password=secure_password \
  --set auth.database=gaap
```

### Manual Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: gaap
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          value: gaap
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: gaap-secrets
              key: POSTGRES_PASSWORD
        - name: POSTGRES_DB
          value: gaap
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

---

## Redis Deployment

```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: gaap
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: 128Mi
            cpu: 100m
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: gaap
spec:
  selector:
    app: redis
  ports:
  - port: 6379
```

---

## Scaling

### Horizontal Scaling

```bash
# Scale manually
kubectl scale deployment gaap-backend --replicas=5 -n gaap

# View HPA status
kubectl get hpa -n gaap

# Watch scaling
kubectl get hpa gaap-backend-hpa -n gaap -w
```

### Vertical Scaling

Edit deployment resources:

```bash
kubectl edit deployment gaap-backend -n gaap
```

Update resources section:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

---

## Monitoring

### Prometheus ServiceMonitor

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gaap-metrics
  namespace: gaap
spec:
  selector:
    matchLabels:
      app: gaap-backend
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### PodMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: gaap-pods
  namespace: gaap
spec:
  selector:
    matchLabels:
      app: gaap-backend
  podMetricsEndpoints:
  - port: http
    path: /metrics
```

---

## Commands

```bash
# Apply all manifests
kubectl apply -f k8s/ -n gaap

# Check pods
kubectl get pods -n gaap

# Check services
kubectl get svc -n gaap

# Check ingress
kubectl get ingress -n gaap

# View logs
kubectl logs -f deployment/gaap-backend -n gaap
kubectl logs -f deployment/gaap-frontend -n gaap

# Execute in pod
kubectl exec -it deployment/gaap-backend -n gaap -- sh

# Port forward for local testing
kubectl port-forward svc/gaap-backend 8000:8000 -n gaap
kubectl port-forward svc/gaap-frontend 3000:3000 -n gaap

# Delete all
kubectl delete -f k8s/ -n gaap
```

---

## Production Checklist

- [ ] Resource limits set
- [ ] Health checks configured
- [ ] HPA enabled
- [ ] Pod disruption budget
- [ ] Network policies
- [ ] Secrets encrypted
- [ ] TLS configured
- [ ] Monitoring enabled
- [ ] Backup strategy
