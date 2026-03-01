# 🔧 Setup Instructions - لتشغيل المشروع

## المتطلبات

### 1. تثبيت Dependencies

```bash
cd /home/fares/Projects/GAAP

# Python dependencies
pip install -e .

# أو إذا عايز تثبيت يدوياً:
pip install sqlalchemy asyncpg aiosqlite alembic redis prometheus-client

# Frontend dependencies
cd frontend
npm install
```

### 2. إعداد Database

```bash
# طريقة 1: Docker (مستحسن)
docker-compose up -d postgres redis

# طريقة 2: تثبيت محلي
# PostgreSQL
sudo apt install postgresql
sudo -u postgres createdb gaap

# Redis
sudo apt install redis-server
```

### 3. تشغيل Migrations

```bash
cd /home/fares/Projects/GAAP
alembic upgrade head
```

### 4. تشغيل المشروع

```bash
# الطريقة السهلة (الكل في واحد)
./start_full_system.sh

# أو يدوياً:

# Terminal 1 - Database
docker-compose up -d postgres redis

# Terminal 2 - Backend
cd /home/fares/Projects/GAAP
python -m gaap.api.main

# Terminal 3 - Frontend
cd /home/fares/Projects/GAAP/frontend
npm run dev

# Terminal 4 - Monitoring (اختياري)
docker-compose -f docker-compose.monitoring.yml up -d
```

## 🌐 URLs بعد التشغيل

| الخدمة | URL |
|--------|-----|
| Web App | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |

## 🧪 اختبار التشغيل

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/api/health

# تشغيل الاختبارات
pytest tests/ -v
```

## ⚠️ ملاحظات مهمة

1. **SQLAlchemy** - المكتبة مش موجودة في البيئة الحالية، لازم تثبتها
2. **Redis** - اختياري للـ cache، المشروع يشتغل بدونه بس أبطأ
3. **PostgreSQL** - ممكن تستخدم SQLite للتطوير لو PostgreSQL مش متاح

## 🆘 في حالة مشاكل

### مشكلة: `ModuleNotFoundError: No module named 'sqlalchemy'`
```bash
pip install sqlalchemy asyncpg
```

### مشكلة: `alembic: command not found`
```bash
pip install alembic
```

### مشكلة: `npm: command not found`
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### مشكلة: Database connection failed
```bash
# تأكد إن PostgreSQL شغال
docker-compose ps

# لو مش شغال
docker-compose up -d postgres
```
