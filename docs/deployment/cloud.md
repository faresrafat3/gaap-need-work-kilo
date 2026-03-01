# Cloud Deployment

Deploy GAAP on major cloud providers.

---

## AWS Deployment

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AWS Cloud                        │
│  ┌───────────────────────────────────────────────┐  │
│  │           Application Load Balancer            │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼────────────────────────────┐  │
│  │              ECS / EKS Cluster                 │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │  │
│  │  │ Frontend│ │ Backend │ │ Backend │         │  │
│  │  │   2x    │ │   3x    │ │   3x    │         │  │
│  │  └─────────┘ └─────────┘ └─────────┘         │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼────────────────────────────┐  │
│  │              RDS PostgreSQL                    │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### ECS Deployment

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name gaap-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json

# Create service
aws ecs create-service \
  --cluster gaap-cluster \
  --service-name gaap-backend \
  --task-definition gaap-backend:1 \
  --desired-count 3 \
  --launch-type FARGATE
```

### ECS Task Definition

```json
{
  "family": "gaap-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "gaap-backend",
      "image": "YOUR_ECR_REPO/gaap-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "GAAP_ENVIRONMENT", "value": "production"}
      ],
      "secrets": [
        {
          "name": "GAAP_KILO_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:gaap/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/gaap-backend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Terraform

```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "gaap-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
}

# ECS Cluster
resource "aws_ecs_cluster" "gaap" {
  name = "gaap-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "gaap" {
  identifier        = "gaap-db"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = "db.t3.medium"
  allocated_storage = 20

  db_name  = "gaap"
  username = "gaap"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.gaap.name
}

# Application Load Balancer
resource "aws_lb" "gaap" {
  name               = "gaap-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
}
```

### Deploy with Terraform

```bash
# Initialize
terraform init

# Plan
terraform plan

# Apply
terraform apply
```

---

## GCP Deployment

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Google Cloud                        │
│  ┌───────────────────────────────────────────────┐  │
│  │           Cloud Load Balancer                  │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼────────────────────────────┐  │
│  │              Cloud Run / GKE                   │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │  │
│  │  │ Frontend│ │ Backend │ │ Backend │         │  │
│  │  └─────────┘ └─────────┘ └─────────┘         │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼────────────────────────────┐  │
│  │            Cloud SQL (PostgreSQL)              │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Cloud Run Deployment

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT/gaap-backend

# Deploy
gcloud run deploy gaap-backend \
  --image gcr.io/PROJECT/gaap-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GAAP_ENVIRONMENT=production \
  --set-secrets GAAP_KILO_API_KEY=api-key:latest

# Deploy frontend
gcloud run deploy gaap-frontend \
  --image gcr.io/PROJECT/gaap-frontend \
  --platform managed \
  --region us-central1 \
  --set-env-vars PYTHON_API_URL=https://gaap-backend-xxx-uc.a.run.app
```

### Cloud Build

```yaml
# cloudbuild.yaml
steps:
  # Build backend
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/gaap-backend', '-f', 'Dockerfile.backend', '.']
  
  # Build frontend
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/gaap-frontend', '-f', 'frontend/Dockerfile', 'frontend/']
  
  # Push images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/gaap-backend']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/gaap-frontend']
  
  # Deploy backend
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'gaap-backend'
      - '--image'
      - 'gcr.io/$PROJECT_ID/gaap-backend'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'

images:
  - 'gcr.io/$PROJECT_ID/gaap-backend'
  - 'gcr.io/$PROJECT_ID/gaap-frontend'
```

### Terraform (GCP)

```hcl
# main.tf
provider "google" {
  project = var.project_id
  region  = "us-central1"
}

# Enable APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
  ])
  service = each.value
}

# Cloud SQL
resource "google_sql_database_instance" "gaap" {
  name             = "gaap-db"
  database_version = "POSTGRES_15"
  region           = "us-central1"

  settings {
    tier = "db-f1-micro"
    
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }
  }
}

resource "google_sql_database" "gaap" {
  name     = "gaap"
  instance = google_sql_database_instance.gaap.name
}

# Cloud Run Backend
resource "google_cloud_run_service" "backend" {
  name     = "gaap-backend"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/gaap-backend"
        
        env {
          name  = "DATABASE_URL"
          value = "postgresql://${var.db_user}:${var.db_password}@/${google_sql_database.gaap.name}?host=/cloudsql/${var.project_id}:us-central1:${google_sql_database_instance.gaap.name}"
        }
      }
    }
  }
}
```

---

## Azure Deployment

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Microsoft Azure                    │
│  ┌───────────────────────────────────────────────┐  │
│  │         Azure Application Gateway              │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼────────────────────────────┐  │
│  │         Azure Container Apps / AKS             │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │  │
│  │  │ Frontend│ │ Backend │ │ Backend │         │  │
│  │  └─────────┘ └─────────┘ └─────────┘         │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼────────────────────────────┐  │
│  │        Azure Database for PostgreSQL           │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Azure Container Apps

```bash
# Create resource group
az group create --name gaap-rg --location eastus

# Create Container Apps environment
az containerapp env create \
  --name gaap-env \
  --resource-group gaap-rg \
  --location eastus

# Deploy backend
az containerapp create \
  --name gaap-backend \
  --resource-group gaap-rg \
  --environment gaap-env \
  --image gaap.azurecr.io/backend:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 2 \
  --max-replicas 10 \
  --env-vars GAAP_ENVIRONMENT=production \
  --secrets "api-key=your-secret-key" \
  --env-vars "GAAP_KILO_API_KEY=secretref:api-key"
```

### Azure Terraform

```hcl
# main.tf
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "gaap" {
  name     = "gaap-rg"
  location = "East US"
}

# Container Registry
resource "azurerm_container_registry" "gaap" {
  name                = "gaapacr"
  resource_group_name = azurerm_resource_group.gaap.name
  location            = azurerm_resource_group.gaap.location
  sku                 = "Standard"
  admin_enabled       = true
}

# Container Apps Environment
resource "azurerm_container_app_environment" "gaap" {
  name                = "gaap-env"
  resource_group_name = azurerm_resource_group.gaap.name
  location            = azurerm_resource_group.gaap.location
}

# Backend Container App
resource "azurerm_container_app" "backend" {
  name                         = "gaap-backend"
  container_app_environment_id = azurerm_container_app_environment.gaap.id
  resource_group_name          = azurerm_resource_group.gaap.name
  revision_mode                = "Single"

  template {
    container {
      name   = "backend"
      image  = "${azurerm_container_registry.gaap.login_server}/backend:latest"
      cpu    = 1.0
      memory = "2Gi"
      
      env {
        name  = "GAAP_ENVIRONMENT"
        value = "production"
      }
    }
    
    min_replicas = 2
    max_replicas = 10
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}

# PostgreSQL
resource "azurerm_postgresql_flexible_server" "gaap" {
  name                   = "gaap-postgres"
  resource_group_name    = azurerm_resource_group.gaap.name
  location               = azurerm_resource_group.gaap.location
  version                = "15"
  administrator_login    = "gaapadmin"
  administrator_password = var.db_password
  
  storage_mb = 32768
  sku_name   = "B_Standard_B1ms"
}
```

---

## Cost Comparison

| Provider | Service | Estimated Monthly Cost |
|----------|---------|----------------------|
| AWS | ECS Fargate + RDS | $200-400 |
| AWS | EKS + RDS | $300-600 |
| GCP | Cloud Run + Cloud SQL | $150-300 |
| GCP | GKE + Cloud SQL | $250-500 |
| Azure | Container Apps + PostgreSQL | $200-400 |
| Azure | AKS + PostgreSQL | $300-600 |

*Costs vary based on usage and region*

---

## Best Practices

### Multi-Cloud Strategy

```hcl
# Use Terraform modules for portability
module "gaap_aws" {
  source = "./modules/gaap"
  provider = "aws"
  # ...
}

module "gaap_gcp" {
  source = "./modules/gaap"
  provider = "gcp"
  # ...
}
```

### Cost Optimization

1. **Use Spot/Preemptible instances** for non-critical workloads
2. **Auto-scaling** based on demand
3. **Reserved capacity** for predictable workloads
4. **CDN** for static assets

### Security

1. **Private subnets** for databases
2. **VPC peering** for secure communication
3. **Secrets management** (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
4. **IAM roles** for service accounts

---

## CI/CD Integration

### GitHub Actions (AWS)

```yaml
# .github/workflows/deploy-aws.yml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build and push
        run: |
          docker build -t $ECR_REGISTRY/gaap-backend:$GITHUB_SHA -f Dockerfile.backend .
          docker push $ECR_REGISTRY/gaap-backend:$GITHUB_SHA
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster gaap-cluster \
            --service gaap-backend --force-new-deployment
```
