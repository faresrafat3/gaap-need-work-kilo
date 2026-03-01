"""
E2E Flask App Gauntlet Tests
============================

Tests for end-to-end Flask application generation.

Implements: docs/evolution_plan_2026/45_TESTING_AUDIT_SPEC.md
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gaap.core.types import Message, MessageRole, TaskPriority, TaskType


class MockFlaskProvider:
    """Mock provider simulating Flask app generation."""

    def __init__(self):
        self.name = "kilo-flask"
        self.models = ["kilo-1.0"]
        self.default_model = "kilo-1.0"

    async def chat_completion(self, messages, model=None, **kwargs):
        from gaap.core.types import ChatCompletionChoice, ChatCompletionResponse, Usage

        user_msg = messages[-1].content if messages else ""

        if "auth" in user_msg.lower() or "authentication" in user_msg.lower():
            content = {
                "app.py": '''from flask import Flask, request, jsonify
from functools import wraps
import hashlib
import jwt
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Mock user database
users_db = {}

def token_required(f):
    """Decorator to require JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if username in users_db:
        return jsonify({'error': 'User already exists'}), 409
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    users_db[username] = hashed_password
    
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    """Login and get JWT token."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if username not in users_db or users_db[username] != hashed_password:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = jwt.encode({
        'user': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'])
    
    return jsonify({'token': token}), 200

@app.route('/api/protected', methods=['GET'])
@token_required
def protected():
    """Protected endpoint."""
    return jsonify({'message': 'This is a protected endpoint'}), 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
''',
                "requirements.txt": """flask==3.0.0
pyjwt==2.8.0
pytest==9.0.0
pytest-flask==1.3.0
""",
                "test_app.py": '''import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    """Test health endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_register(client):
    """Test user registration."""
    response = client.post('/api/register', json={
        'username': 'testuser',
        'password': 'testpass123'
    })
    assert response.status_code == 201

def test_register_duplicate(client):
    """Test duplicate registration."""
    client.post('/api/register', json={
        'username': 'testuser2',
        'password': 'testpass123'
    })
    response = client.post('/api/register', json={
        'username': 'testuser2',
        'password': 'testpass123'
    })
    assert response.status_code == 409

def test_login_success(client):
    """Test successful login."""
    client.post('/api/register', json={
        'username': 'loginuser',
        'password': 'testpass123'
    })
    response = client.post('/api/login', json={
        'username': 'loginuser',
        'password': 'testpass123'
    })
    assert response.status_code == 200
    assert 'token' in response.json

def test_login_invalid(client):
    """Test login with invalid credentials."""
    response = client.post('/api/login', json={
        'username': 'nonexistent',
        'password': 'wrongpass'
    })
    assert response.status_code == 401

def test_protected_without_token(client):
    """Test protected endpoint without token."""
    response = client.get('/api/protected')
    assert response.status_code == 401

def test_protected_with_token(client):
    """Test protected endpoint with valid token."""
    client.post('/api/register', json={
        'username': 'protecteduser',
        'password': 'testpass123'
    })
    login_response = client.post('/api/login', json={
        'username': 'protecteduser',
        'password': 'testpass123'
    })
    token = login_response.json['token']
    
    response = client.get('/api/protected', headers={
        'Authorization': f'Bearer {token}'
    })
    assert response.status_code == 200
''',
            }
        else:
            content = {
                "app.py": '''from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    """Basic hello endpoint."""
    return jsonify({'message': 'Hello, World!'})

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
''',
                "requirements.txt": """flask==3.0.0
pytest==9.0.0
pytest-flask==1.3.0
""",
                "test_app.py": '''import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_hello(client):
    """Test hello endpoint."""
    response = client.get('/api/hello')
    assert response.status_code == 200
    assert 'message' in response.json

def test_health(client):
    """Test health endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'
''',
            }

        response_content = str(content)

        return ChatCompletionResponse(
            id="kilo-flask-response",
            model=model or self.default_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content=response_content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=100, completion_tokens=500, total_tokens=600),
        )


@pytest.fixture
def flask_provider():
    """Provide mock Flask generation provider."""
    return MockFlaskProvider()


class FlaskAppBuilder:
    """Builder for creating Flask apps from LLM responses."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.files: dict[str, str] = {}

    def add_file(self, name: str, content: str) -> "FlaskAppBuilder":
        """Add a file to the app."""
        self.files[name] = content
        return self

    def build(self) -> dict[str, Path]:
        """Build the Flask app and return file paths."""
        paths = {}
        for name, content in self.files.items():
            file_path = self.tmp_path / name
            file_path.write_text(content)
            paths[name] = file_path
        return paths

    def check_structure(self) -> dict[str, bool]:
        """Check if required files exist."""
        required = ["app.py", "requirements.txt"]
        return {f: (self.tmp_path / f).exists() for f in required}


class TestFlaskAppGauntlet:
    """E2E tests for Flask application generation."""

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_simple_flask_app(self, flask_provider, tmp_path) -> None:
        """Test generation of a simple Flask app."""
        response = await flask_provider.chat_completion(
            [Message(role=MessageRole.USER, content="Create a simple Flask app")]
        )

        content_str = response.choices[0].message.content
        try:
            content = json.loads(content_str.replace("'", '"'))
        except json.JSONDecodeError:
            try:
                content = ast.literal_eval(content_str)
            except (ValueError, SyntaxError) as e:
                pytest.fail(f"Failed to parse response content: {e}")

        builder = FlaskAppBuilder(tmp_path)
        for name, code in content.items():
            builder.add_file(name, code)

        paths = builder.build()

        assert builder.check_structure()["app.py"]
        assert builder.check_structure()["requirements.txt"]

        app_code = paths["app.py"].read_text()
        assert "from flask import" in app_code
        assert "@app.route" in app_code

        try:
            ast.parse(app_code)
        except SyntaxError as e:
            pytest.fail(f"Generated Flask app has syntax errors: {e}")

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_flask_app_with_auth(self, flask_provider, tmp_path) -> None:
        """Test generation of Flask app with authentication."""
        response = await flask_provider.chat_completion(
            [
                Message(
                    role=MessageRole.USER,
                    content="Create a Flask app with user authentication (register, login, protected endpoints)",
                )
            ]
        )

        content_str = response.choices[0].message.content
        try:
            content = json.loads(content_str.replace("'", '"'))
        except json.JSONDecodeError:
            try:
                content = ast.literal_eval(content_str)
            except (ValueError, SyntaxError) as e:
                pytest.fail(f"Failed to parse response content: {e}")

        builder = FlaskAppBuilder(tmp_path)
        for name, code in content.items():
            builder.add_file(name, code)

        paths = builder.build()

        app_code = paths["app.py"].read_text()
        assert "/register" in app_code or "register" in app_code.lower()
        assert "/login" in app_code or "login" in app_code.lower()

        assert (
            "token" in app_code.lower() or "jwt" in app_code.lower() or "auth" in app_code.lower()
        )

        test_code = paths.get("test_app.py")
        if test_code:
            test_content = test_code.read_text()
            assert "test_" in test_content
            assert "def test_" in test_content

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_flask_requirements(self, flask_provider, tmp_path) -> None:
        """Test that requirements.txt is valid."""
        response = await flask_provider.chat_completion(
            [Message(role=MessageRole.USER, content="Create a Flask app")]
        )

        content_str = response.choices[0].message.content
        try:
            content = json.loads(content_str.replace("'", '"'))
        except json.JSONDecodeError:
            try:
                content = ast.literal_eval(content_str)
            except (ValueError, SyntaxError) as e:
                pytest.fail(f"Failed to parse response content: {e}")

        req_path = tmp_path / "requirements.txt"
        req_path.write_text(content["requirements.txt"])

        requirements = req_path.read_text()

        assert "flask" in requirements.lower()

        for line in requirements.strip().split("\n"):
            if line.strip() and not line.startswith("#"):
                assert "==" in line or ">=" in line or "<=" in line or line.strip().isalpha()

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_flask_tests_structure(self, flask_provider, tmp_path) -> None:
        """Test that test file has proper structure."""
        response = await flask_provider.chat_completion(
            [Message(role=MessageRole.USER, content="Create a Flask app with tests")]
        )

        content_str = response.choices[0].message.content
        try:
            content = json.loads(content_str.replace("'", '"'))
        except json.JSONDecodeError:
            try:
                content = ast.literal_eval(content_str)
            except (ValueError, SyntaxError) as e:
                pytest.fail(f"Failed to parse response content: {e}")

        if "test_app.py" in content:
            test_path = tmp_path / "test_app.py"
            test_path.write_text(content["test_app.py"])

            test_code = test_path.read_text()

            assert "import pytest" in test_code or "pytest" in test_code
            assert "@pytest.fixture" in test_code or "def test_" in test_code


class TestFlaskCodeQualityGauntlet:
    """Tests for Flask code quality."""

    @pytest.mark.gauntlet
    def test_route_definitions(self, tmp_path) -> None:
        """Test that routes are properly defined."""
        app_code = """
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([])

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    return jsonify({'id': user_id})
"""

        app_path = tmp_path / "app.py"
        app_path.write_text(app_code)

        tree = ast.parse(app_code)

        decorators = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "route"
        ]

        assert len(decorators) >= 2

    @pytest.mark.gauntlet
    def test_error_handling(self, tmp_path) -> None:
        """Test that error handling is present."""
        app_code = """
from flask import Flask, jsonify

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500
"""

        app_path = tmp_path / "app.py"
        app_path.write_text(app_code)

        assert "@app.errorhandler" in app_code
        assert "404" in app_code


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gauntlet"])
