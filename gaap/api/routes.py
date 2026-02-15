# API Routes
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from gaap.gaap_engine import GAAPEngine, GAAPRequest, create_engine

# =============================================================================
# Simple HTTP Server (without external dependencies)
# =============================================================================

class Request:
    """طلب HTTP"""
    def __init__(self, method: str, path: str, headers: dict[str, str], body: str = ""):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body

    def json(self) -> dict[str, Any]:
        try:
            return json.loads(self.body)
        except:
            return {}


class Response:
    """استجابة HTTP"""
    def __init__(self, status: int = 200, body: Any = None, headers: dict[str, str] = None):
        self.status = status
        self.body = body or {}
        self.headers = headers or {"Content-Type": "application/json"}

    def to_bytes(self) -> bytes:
        body_str = json.dumps(self.body, ensure_ascii=False, indent=2)
        headers_str = "\r\n".join(f"{k}: {v}" for k, v in self.headers.items())

        response_str = f"HTTP/1.1 {self.status} OK\r\n"
        response_str += headers_str + "\r\n"
        response_str += f"Content-Length: {len(body_str.encode())}\r\n"
        response_str += "\r\n"
        response_str += body_str

        return response_str.encode()


# =============================================================================
# API Router
# =============================================================================

class GAPAPIRouter:
    """موجه API"""

    def __init__(self, engine: GAAPEngine):
        self.engine = engine
        self._logger = logging.getLogger("gaap.api")

    async def handle(self, request: Request) -> Response:
        """معالجة الطلب"""
        try:
            # التوجيه
            if request.method == "GET":
                if request.path == "/health":
                    return await self._health()
                elif request.path == "/status":
                    return await self._status()
                elif request.path == "/":
                    return Response(body={"message": "GAAP API v1.0.0", "endpoints": ["/chat", "/execute", "/status", "/health"]})

            elif request.method == "POST":
                if request.path == "/chat":
                    return await self._chat(request)
                elif request.path == "/execute":
                    return await self._execute(request)
                elif request.path == "/batch":
                    return await self._batch(request)

            return Response(status=404, body={"error": "Not found"})

        except Exception as e:
            self._logger.error(f"Error handling request: {e}")
            return Response(status=500, body={"error": str(e)})

    async def _health(self) -> Response:
        """فحص الصحة"""
        return Response(body={
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })

    async def _status(self) -> Response:
        """حالة النظام"""
        stats = self.engine.get_stats()
        return Response(body={
            "status": "running",
            "stats": stats
        })

    async def _chat(self, request: Request) -> Response:
        """محادثة سريعة"""
        data = request.json()

        if "message" not in data:
            return Response(status=400, body={"error": "Missing 'message' field"})

        start = time.time()
        response_text = await self.engine.chat(data["message"])
        elapsed = (time.time() - start) * 1000

        return Response(body={
            "response": response_text,
            "time_ms": elapsed
        })

    async def _execute(self, request: Request) -> Response:
        """تنفيذ مهمة كاملة"""
        data = request.json()

        if "task" not in data:
            return Response(status=400, body={"error": "Missing 'task' field"})

        gaap_request = GAAPRequest(
            text=data["task"],
            budget_limit=data.get("budget", 10.0)
        )

        response = await self.engine.process(gaap_request)

        return Response(body={
            "success": response.success,
            "output": response.output,
            "error": response.error,
            "metrics": {
                "time_ms": response.total_time_ms,
                "cost_usd": response.total_cost_usd,
                "tokens": response.total_tokens,
                "quality_score": response.quality_score
            },
            "details": {
                "intent": response.intent.intent_type.name if response.intent else None,
                "tasks_executed": len(response.execution_results)
            }
        })

    async def _batch(self, request: Request) -> Response:
        """تنفيذ مجموعة مهام"""
        data = request.json()

        if "tasks" not in data or not isinstance(data["tasks"], list):
            return Response(status=400, body={"error": "Missing 'tasks' list"})

        results = []
        for task_text in data["tasks"][:10]:  # حد 10 مهام
            request = GAAPRequest(text=task_text)
            response = await self.engine.process(request)
            results.append({
                "task": task_text[:50],
                "success": response.success,
                "output": str(response.output)[:200] if response.output else None,
                "error": response.error
            })

        return Response(body={
            "total": len(results),
            "results": results
        })


# =============================================================================
# HTTP Server
# =============================================================================

class GAAPAPIServer:
    """خادم API"""

    def __init__(self, engine: GAAPEngine, host: str = "0.0.0.0", port: int = 8080):
        self.engine = engine
        self.host = host
        self.port = port
        self.router = GAPAPIRouter(engine)
        self._logger = logging.getLogger("gaap.api.server")

    async def start(self):
        """بدء الخادم"""
        import asyncio

        server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port
        )

        self._logger.info(f"GAAP API Server running on http://{self.host}:{self.port}")

        async with server:
            await server.serve_forever()

    async def _handle_connection(self, reader, writer):
        """معالجة الاتصال"""
        try:
            # قراءة الطلب
            data = await reader.read(65536)
            request_str = data.decode()

            # تحليل الطلب
            request = self._parse_request(request_str)

            # معالجة
            response = await self.router.handle(request)

            # إرسال الرد
            writer.write(response.to_bytes())
            await writer.drain()

        except Exception as e:
            self._logger.error(f"Connection error: {e}")
        finally:
            writer.close()

    def _parse_request(self, request_str: str) -> Request:
        """تحليل الطلب"""
        lines = request_str.split("\r\n")

        # السطر الأول
        first_line = lines[0].split(" ")
        method = first_line[0]
        path = first_line[1]

        # Headers
        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line == "":
                body_start = i + 1
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()

        # Body
        body = "\r\n".join(lines[body_start:]) if body_start < len(lines) else ""

        return Request(method, path, headers, body)


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_server(host: str = "0.0.0.0", port: int = 8080, budget: float = 100.0):
    """تشغيل الخادم"""
    # إنشاء المحرك
    engine = create_engine(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        budget=budget
    )

    # إنشاء وتشغيل الخادم
    server = GAAPAPIServer(engine, host, port)
    await server.start()


def main():
    """نقطة الدخول"""
    import argparse

    parser = argparse.ArgumentParser(description="GAAP API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--budget", type=float, default=100.0)

    args = parser.parse_args()

    # إعداد التسجيل
    logging.basicConfig(level=logging.INFO)

    # تشغيل
    asyncio.run(run_server(args.host, args.port, args.budget))


if __name__ == "__main__":
    main()
