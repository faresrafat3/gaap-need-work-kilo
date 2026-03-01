# GAAP Troubleshooting Guide

## Common Issues

### 🔴 Frontend Can't Connect to Backend

**Symptoms:**
- Error: "Backend unreachable"
- Provider status shows "offline"

**Solutions:**
1. Verify backend is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check `PYTHON_API_URL` in `.env.local`:
   ```bash
   # For local development
   PYTHON_API_URL=http://localhost:8000
   
   # For Docker
   PYTHON_API_URL=http://gaap-backend:8000
   ```

3. Check CORS settings on backend

---

### 🔴 Build Fails

**Symptoms:**
- `npm run build` fails
- TypeScript errors

**Solutions:**
1. Clear cache:
   ```bash
   rm -rf node_modules .next
   npm install
   ```

2. Check TypeScript:
   ```bash
   npx tsc --noEmit
   ```

3. Check for syntax errors in your code

---

### 🔴 Rate Limit Exceeded (429 Error)

**Symptoms:**
- "تم تجاوز الحد الأقصى للطلبات"
- Requests blocked

**Solutions:**
1. Wait for rate limit reset (1 minute)
2. Increase limit in `.env.local`:
   ```bash
   RATE_LIMIT_REQUESTS_PER_MINUTE=120
   ```
3. Check for infinite loops in your code

---

### 🔴 Session Not Found

**Symptoms:**
- "الجلسة غير موجودة" error
- Can't load session data

**Solutions:**
1. Verify session ID is correct
2. Check backend database connection
3. Clear browser localStorage and retry

---

### 🔴 Provider Shows "Offline"

**Symptoms:**
- Provider status is offline
- Can't select provider

**Solutions:**
1. Check provider configuration in backend
2. Verify provider credentials
3. Check provider's actual service status
4. Restart backend to refresh provider cache

---

### 🔴 Chat Not Responding

**Symptoms:**
- Messages not sending
- No response from AI

**Solutions:**
1. Check browser console for errors
2. Verify backend is processing requests
3. Check provider is selected and online
4. Clear chat history and retry

---

## Performance Issues

### Slow Loading

**Solutions:**
1. Enable production build
2. Check network connection
3. Reduce number of providers being fetched
4. Enable caching

### High Memory Usage

**Solutions:**
1. Clear old sessions
2. Reduce cache size
3. Restart the application

## Getting Help

1. Check logs: `docker-compose logs -f`
2. Enable debug mode: `NODE_ENV=development`
3. Check health endpoint: `/api/health`
4. Review API documentation: `API.md`
