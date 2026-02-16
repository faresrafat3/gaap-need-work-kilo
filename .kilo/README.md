# Kilo Knowledge System

## Quick Commands

| Command | Description |
|---------|-------------|
| `/compact` | حفظ المعرفة وتنظيف السياق |
| `/knowledge add` | إضافة درس/حل جديد |
| `/knowledge search <query>` | بحث في المعرفة |
| `/knowledge list` | عرض المعرفة |
| `/knowledge stats` | إحصائيات |
| `/session save` | حفظ الجلسة |
| `/wisdom` | أهم الدروس |

## Auto-Compact Behavior

- **70% context**: حفظ تلقائي صامت
- **85% context**: تحذير + اقتراح /compact
- **نهاية الجلسة**: حفظ تلقائي كامل

## Knowledge Storage

```
~/.kilo/knowledge/         # معرفة عامة
.kilo/knowledge/           # معرفة المشروع
```

## Current Lessons (Top 3)

1. **لا تستخدم `# mypy: ignore-errors`** - حل سطحي
2. **تحويل float إلى int صراحة**: `int(len(x) * 1.5)`
3. **إضافة return type لكل دالة**: `def foo() -> None:`

## Anti-Patterns to Avoid

1. `# mypy: ignore-errors` - خطير!
2. دوال بدون type annotations
3. إفراط في `# type: ignore`
