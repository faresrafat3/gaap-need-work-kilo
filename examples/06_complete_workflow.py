"""
GAAP Complete Workflow Example

هذا المثال يوضح سيناريو حقيقي متكامل لاستخدام GAAP:
1. إنشاء مشروع جديد من الصفر
2. تنفيذ مهام متعددة الطبقات
3. استخدام الذاكرة للتعلم من الأخطاء
4. التعامل مع الأخطاء والتعافي الذاتي
5. مراقبة الإحصائيات والأداء

المطلوب:
- API Keys في ملف .gaap_env
- بيئة Python 3.10+
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any

# إضافة مسار المشروع
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaap import GAAPEngine, GAAPRequest, create_engine
from gaap.core.types import TaskPriority, TaskType
from gaap.memory.hierarchical import HierarchicalMemory, EpisodicMemory


# =============================================================================
# Helper Functions
# =============================================================================

def print_section(title: str, char: str = "=") -> None:
    """طباعة عنوان قسم"""
    width = 70
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_response(response: Any, show_output: bool = True) -> None:
    """طباعة استجابة GAAP"""
    print(f"\n✅ Success: {response.success}")
    print(f"⏱️  Time: {response.total_time_ms:.0f}ms ({response.total_time_ms/1000:.2f}s)")
    print(f"💰 Cost: ${response.total_cost_usd:.4f}")
    print(f"🎯 Quality Score: {response.quality_score:.2f}/1.00")
    print(f"🔄 OODA Iterations: {response.ooda_iterations}")
    print(f"📊 Replans: {response.strategic_replan_count}")
    print(f"⚠️  Axiom Violations: {response.axiom_violation_count}")
    print(f"📝 Tasks Executed: {len(response.execution_results)}")
    
    if show_output and response.output:
        print("\n📄 Output Preview:")
        print("-" * 70)
        # عرض أول 500 حرف فقط
        output_preview = str(response.output)[:500]
        print(output_preview)
        if len(str(response.output)) > 500:
            print(f"\n... ({len(str(response.output)) - 500} more characters)")
        print("-" * 70)


# =============================================================================
# Example 1: Simple Code Generation
# =============================================================================

async def example_1_simple_code() -> None:
    """مثال 1: توليد كود بسيط"""
    print_section("Example 1: Simple Code Generation")
    
    engine = GAAPEngine(
        budget=5.0,
        enable_healing=True,
        enable_memory=True,
        enable_security=True,
    )
    
    request = GAAPRequest(
        text="""
Write a Python function that implements binary search on a sorted array.
The function should:
- Take a sorted list and a target value
- Return the index of the target if found, -1 otherwise
- Handle edge cases (empty list, not found)
- Include type hints and docstring
- Add example usage
""",
        priority=TaskPriority.NORMAL,
    )
    
    print("\n📝 Request: Write a binary search function...")
    response = await engine.process(request)
    
    print_response(response)
    
    # حفظ الدرس في الذاكرة
    if response.success and engine.memory:
        episode = EpisodicMemory(
            task_id="binary_search_example",
            action="code_generation",
            result="Successfully generated binary search with edge cases",
            success=True,
            duration_ms=response.total_time_ms,
            tokens_used=response.total_tokens,
            cost_usd=response.total_cost_usd,
            model="llama-3.3-70b",
            provider="groq",
            lessons=[
                "Binary search requires careful handling of mid calculation",
                "Edge cases: empty list, single element, not found"
            ]
        )
        engine.memory.record_episode(episode)
        print("\n💾 Episode recorded to memory")
    
    engine.shutdown()


# =============================================================================
# Example 2: Multi-Step Project Creation
# =============================================================================

async def example_2_project_creation() -> None:
    """مثال 2: إنشاء مشروع متكامل"""
    print_section("Example 2: Multi-Step Project Creation")
    
    engine = create_engine(
        budget=20.0,
        enable_all=True,
    )
    
    # المهمة 1: التخطيط
    print("\n🎯 Step 1: Planning the project architecture...")
    planning_request = GAAPRequest(
        text="""
Design a REST API for a task management system with:
- User authentication (JWT)
- CRUD operations for tasks
- Task categories and tags
- Due dates and priorities
- PostgreSQL database
- FastAPI framework

Provide:
1. Project structure
2. Database schema
3. API endpoints
4. Authentication flow
""",
        priority=TaskPriority.HIGH,
    )
    
    planning_response = await engine.process(planning_request)
    print_response(planning_response, show_output=False)
    
    # المهمة 2: كتابة الكود
    print("\n🎯 Step 2: Implementing the core models...")
    code_request = GAAPRequest(
        text="""
Based on the architecture above, implement the SQLAlchemy models for:
- User model with password hashing
- Task model with relationships
- Category and Tag models
- Include validation and constraints
""",
        priority=TaskPriority.NORMAL,
        context={"architecture": planning_response.output} if planning_response.output else None,
    )
    
    code_response = await engine.process(code_request)
    print_response(code_response)
    
    # المهمة 3: كتابة الاختبارات
    print("\n🎯 Step 3: Writing unit tests...")
    test_request = GAAPRequest(
        text="""
Write pytest unit tests for the models:
- Test User creation and password hashing
- Test Task CRUD operations
- Test relationships and cascades
- Test validation constraints
""",
        priority=TaskPriority.NORMAL,
    )
    
    test_response = await engine.process(test_request)
    print_response(test_response, show_output=False)
    
    # عرض الإحصائيات
    print("\n📊 Project Statistics:")
    stats = engine.get_stats()
    print(f"  Total Requests: {stats['requests_processed']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Total Cost: ${sum(r.get('cost', 0) for r in stats.get('layer3_stats', {}).values()):.4f}")
    
    engine.shutdown()


# =============================================================================
# Example 3: Code Review and Refactoring
# =============================================================================

async def example_3_code_review() -> None:
    """مثال 3: مراجعة وتحسين الكود"""
    print_section("Example 3: Code Review and Refactoring")
    
    engine = GAAPEngine(
        budget=10.0,
        enable_axiom_enforcement=True,
    )
    
    # كود به مشاكل للمراجعة
    problematic_code = """
def calc(a,b):
    result=a/b
    return result

def process_list(lst):
    new_lst=[]
    for i in range(len(lst)):
        if lst[i]%2==0:
            new_lst.append(lst[i]*2)
        else:
            new_lst.append(lst[i])
    return new_lst

def read_file(path):
    f=open(path,'r')
    data=f.read()
    f.close()
    return data
"""
    
    print("\n🔍 Step 1: Code Review...")
    review_request = GAAPRequest(
        text=f"""
Review this Python code for:
- Code style and best practices
- Error handling
- Security issues
- Performance improvements
- Type hints

Code:
{problematic_code}

Provide specific suggestions for improvement.
""",
        priority=TaskPriority.NORMAL,
    )
    
    review_response = await engine.process(review_request)
    print_response(review_response)
    
    print("\n✨ Step 2: Refactoring based on review...")
    refactor_request = GAAPRequest(
        text="""
Refactor the code based on the review above:
- Add proper error handling
- Add type hints
- Follow PEP 8 style
- Use context managers for files
- Add docstrings
- Improve variable names
""",
        priority=TaskPriority.NORMAL,
        context={"review": review_response.output} if review_response.output else None,
    )
    
    refactor_response = await engine.process(refactor_request)
    print_response(refactor_response)
    
    engine.shutdown()


# =============================================================================
# Example 4: Debugging with Self-Healing
# =============================================================================

async def example_4_debugging() -> None:
    """مثال 4: تصحيح الأخطاء مع التعافي الذاتي"""
    print_section("Example 4: Debugging with Self-Healing")
    
    engine = GAAPEngine(
        budget=10.0,
        enable_healing=True,
        enable_memory=True,
    )
    
    # كود به خطأ
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# Test
print(calculate_average([1, 2, 3, 4, 5]))
print(calculate_average([]))  # This will cause ZeroDivisionError
"""
    
    print("\n🐛 Step 1: Identify the bug...")
    debug_request = GAAPRequest(
        text=f"""
Analyze this code and identify potential bugs:

{buggy_code}

Explain:
1. What's the bug?
2. When does it occur?
3. What's the impact?
""",
        priority=TaskPriority.HIGH,
    )
    
    debug_response = await engine.process(debug_request)
    print_response(debug_response)
    
    print("\n🔧 Step 2: Fix the bug...")
    fix_request = GAAPRequest(
        text="""
Fix the identified bug and add proper error handling.
Also add input validation and edge case handling.
""",
        priority=TaskPriority.NORMAL,
        context={"bug_analysis": debug_response.output} if debug_response.output else None,
    )
    
    fix_response = await engine.process(fix_request)
    print_response(fix_response)
    
    # عرض سجل التعافي
    if engine.healing_system:
        print("\n📋 Healing Records:")
        for record in engine.healing_system.get_healing_history()[-5:]:
            print(f"  - {record.task_id}: {record.level.name} - {'✅' if record.success else '❌'}")
    
    engine.shutdown()


# =============================================================================
# Example 5: Memory and Learning
# =============================================================================

async def example_5_memory_learning() -> None:
    """مثال 5: الذاكرة والتعلم"""
    print_section("Example 5: Memory and Learning from Experience")
    
    memory = HierarchicalMemory()
    
    # تسجيل أحداث سابقة
    print("\n📚 Recording past episodes...")
    
    episodes = [
        EpisodicMemory(
            task_id="task_001",
            action="code_generation",
            result="Generated REST API with authentication",
            success=True,
            duration_ms=3500,
            tokens_used=4500,
            cost_usd=0.15,
            model="llama-3.3-70b",
            provider="groq",
            lessons=["Use JWT for stateless authentication", "Validate all inputs"]
        ),
        EpisodicMemory(
            task_id="task_002",
            action="debugging",
            result="Fixed SQL injection vulnerability",
            success=True,
            duration_ms=2100,
            tokens_used=2800,
            cost_usd=0.08,
            model="claude-3-5-sonnet",
            provider="anthropic",
            lessons=["Always use parameterized queries", "Sanitize user inputs"]
        ),
        EpisodicMemory(
            task_id="task_003",
            action="code_review",
            result="Identified performance bottleneck",
            success=True,
            duration_ms=1800,
            tokens_used=2200,
            cost_usd=0.05,
            model="gpt-4o-mini",
            provider="openai",
            lessons=["Use indexing for database queries", "Cache frequently accessed data"]
        ),
    ]
    
    for episode in episodes:
        memory.record_episode(episode)
        print(f"  ✓ Recorded: {episode.task_id}")
    
    # البحث عن دروس
    print("\n🔍 Searching for relevant lessons...")
    
    search_queries = [
        "authentication security",
        "database performance",
        "input validation"
    ]
    
    for query in search_queries:
        lessons = memory.search_lessons(query, top_k=2)
        print(f"\n  Query: '{query}'")
        print(f"  Found {len(lessons)} relevant lessons")
        for lesson in lessons:
            print(f"    - {lesson.get('lessons', ['N/A'])[0] if lesson.get('lessons') else 'N/A'}")
    
    # عرض إحصائيات الذاكرة
    print("\n📊 Memory Statistics:")
    stats = memory.get_stats()
    print(f"  Total Episodes: {stats.get('total_episodes', 0)}")
    print(f"  Working Memory Items: {stats.get('working_memory_size', 0)}")
    print(f"  Semantic Rules: {stats.get('semantic_rules_count', 0)}")
    
    # استخراج قواعد دلالية
    print("\n🧠 Extracting semantic rules...")
    memory.consolidate_episodes_to_semantic()
    rules = memory.get_semantic_rules()
    print(f"  Extracted {len(rules)} semantic rules")
    for rule in rules[:3]:
        print(f"    - {rule.condition[:50]}... → {rule.action[:50]}...")


# =============================================================================
# Example 6: Provider Comparison
# =============================================================================

async def example_6_provider_comparison() -> None:
    """مثال 6: مقارنة المزودين"""
    print_section("Example 6: Provider Comparison")
    
    from gaap.providers import GroqProvider, GeminiProvider
    
    # الحصول على المفاتيح من البيئة
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_keys_raw = os.getenv("GEMINI_API_KEYS", "")
    gemini_keys = [k.strip() for k in gemini_keys_raw.split(",") if k.strip()]
    
    if not groq_key or not gemini_keys:
        print("⚠️  Skipping: API keys not configured")
        print("  Set GROQ_API_KEY and GEMINI_API_KEYS environment variables")
        return
    
    providers = {
        "Groq": GroqProvider(api_key=groq_key),
        "Gemini": GeminiProvider(api_key=gemini_keys[0], api_keys=gemini_keys),
    }
    
    test_prompt = "Write a Python function to check if a string is a palindrome"
    
    results = {}
    
    for name, provider in providers.items():
        print(f"\n🚀 Testing {name}...")
        try:
            from gaap.core.types import Message, MessageRole
            
            messages = [
                Message(role=MessageRole.USER, content=test_prompt)
            ]
            
            start = datetime.now()
            response = await provider.chat_completion(
                messages=messages,
                model=provider.get_available_models()[0] if provider.get_available_models() else "default"
            )
            elapsed = (datetime.now() - start).total_seconds()
            
            results[name] = {
                "success": True,
                "latency_ms": elapsed * 1000,
                "tokens": response.usage.total_tokens if response.usage else 0,
                "content_length": len(response.choices[0].message.content) if response.choices else 0,
            }
            
            print(f"  ✅ Success")
            print(f"  ⏱️  Latency: {results[name]['latency_ms']:.0f}ms")
            print(f"  📝 Tokens: {results[name]['tokens']}")
            
        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e),
            }
            print(f"  ❌ Failed: {e}")
    
    # مقارنة النتائج
    print("\n📊 Provider Comparison:")
    print(f"{'Provider':<15} {'Status':<10} {'Latency':<15} {'Tokens':<10}")
    print("-" * 50)
    for name, result in results.items():
        status = "✅ Success" if result["success"] else f"❌ {result.get('error', 'Failed')}"
        latency = f"{result.get('latency_ms', 0):.0f}ms" if result["success"] else "N/A"
        tokens = str(result.get("tokens", 0)) if result["success"] else "N/A"
        print(f"{name:<15} {status:<10} {latency:<15} {tokens:<10}")


# =============================================================================
# Main Execution
# =============================================================================

async def run_all_examples() -> None:
    """تشغيل جميع الأمثلة"""
    print_section("🚀 GAAP Complete Workflow Examples", char="═")
    
    examples = [
        ("Simple Code Generation", example_1_simple_code),
        ("Project Creation", example_2_project_creation),
        ("Code Review", example_3_code_review),
        ("Debugging", example_4_debugging),
        ("Memory & Learning", example_5_memory_learning),
        ("Provider Comparison", example_6_provider_comparison),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(2)  # تأخير بين الأمثلة
        except Exception as e:
            print(f"\n❌ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print_section("✅ All Examples Completed", char="═")


def main() -> None:
    """النقطة الرئيسية للدخول"""
    print("\n" + "=" * 70)
    print("  GAAP - General-purpose AI Architecture Platform")
    print("  Complete Workflow Examples")
    print("=" * 70)
    
    # التحقق من المفاتيح
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_keys = os.getenv("GEMINI_API_KEYS", "")
    
    if not groq_key and not gemini_keys:
        print("\n⚠️  WARNING: No API keys configured!")
        print("  Create a .gaap_env file with:")
        print("    GROQ_API_KEY=gsk_...")
        print("    GEMINI_API_KEYS=key1,key2,...")
        print("\n  Some examples may not work without API keys.\n")
    
    # تشغيل الأمثلة
    asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
