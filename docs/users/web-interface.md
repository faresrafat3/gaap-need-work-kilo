# GAAP Web Interface Guide

Complete guide to using GAAP's web interface.

## Overview

The GAAP web interface provides:
- Real-time chat with AI providers
- Session management
- OODA loop visualization
- Provider monitoring
- Usage analytics

## Accessing the Interface

### Local Development

```
http://localhost:3000
```

### Production

```
https://your-domain.com
```

## Main Components

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                    │ Main Content                   │
│  ─────────                  │ ────────────────────────────   │
│                             │                                │
│  Providers                  │ Chat Interface                │
│  ├── Kimi ●                 │ ┌─────────────────────────┐     │
│  ├── DeepSeek ●             │ │ Message History         │     │
│  └── GLM ○                  │ ├─────────────────────────┤     │
│                             │ │ User: Hello!            │     │
│  Sessions                   │ │ AI: Hi! How can I help? │     │
│  ├── Active (3)             │ └─────────────────────────┘     │
│  ├── Paused (1)             │                                │
│  └── Archived (12)          │ Input Box                      │
│                             │ [Type message...    ] [Send]   │
│  System                     │                                │
│  ├── Health ●               │ Status: Connected ●            │
│  └── Metrics                │                                │
│                             └────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Chat Interface

### Sending Messages

1. Type in the input box
2. Press Enter or click Send
3. Watch the response stream in real-time

### Features

- **Markdown support**: Code blocks, lists, formatting
- **Code highlighting**: Syntax highlighted code blocks
- **Streaming**: Real-time response streaming
- **History**: Conversation history within session

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift + Enter` | New line |
| `Ctrl + K` | Clear chat |
| `Ctrl + /` | Show shortcuts |

## Sessions

### Creating a Session

1. Click "New Session" button
2. Enter session name
3. Set priority (optional)
4. Add tags (optional)

### Session States

| State | Icon | Description |
|-------|------|-------------|
| Active | ● Green | Currently running |
| Paused | ⏸ Yellow | Temporarily paused |
| Completed | ✓ Blue | Successfully finished |
| Failed | ✗ Red | Error occurred |
| Archived | 🗑 Gray | Deleted/hidden |

### Managing Sessions

- **Pause**: Stop current execution
- **Resume**: Continue paused session
- **Export**: Download session data (JSON)
- **Archive**: Hide from active list

### Session Details

Click a session to see:
- Full conversation history
- Token usage stats
- Cost breakdown
- Execution timeline

## Providers Panel

### Provider Status

Green dot = Available
Red dot = Unavailable
Gray dot = Disabled

### Selecting Provider

1. Click provider in sidebar
2. Or use dropdown in chat input
3. GAAP automatically falls back if selected fails

### Provider Details

Hover over provider to see:
- Current model
- Success rate
- Average latency
- Total requests

## OODA Visualization

### Real-time OODA Loop

When GAAP processes a request:

```
┌─────────┐
│ Observe │ ◄── Analyzing input...
└────┬────┘
     │
     ▼
┌─────────┐
│ Orient  │     Planning approach...
└────┬────┘
     │
     ▼
┌─────────┐
│ Decide  │     Breaking into tasks...
└────┬────┘
     │
     ▼
┌─────────┐
│   Act   │     Executing...
└─────────┘
```

### Viewing OODA State

- Open OODA panel from sidebar
- Watch phase transitions in real-time
- See task breakdown during "Decide" phase
- View execution progress during "Act"

## System Dashboard

### Health Status

Overview of system components:
- API status
- Database connection
- Provider health
- Memory usage

### Metrics

Real-time charts:
- Request rate
- Response time
- Token usage
- Error rate
- Active connections

### Budget Tracking

```
Daily Usage: $45.50 / $200.00 (22%)
Monthly Usage: $1,200 / $5,000 (24%)

Usage by Provider:
Kimi:        ████████████ 60%
DeepSeek:    ██████ 30%
GLM:         ██ 10%
```

## Settings

### Interface Settings

- **Theme**: Light / Dark / Auto
- **Language**: English / Arabic
- **Font size**: Small / Medium / Large
- **Animations**: On / Off

### Chat Settings

- **Auto-send**: Enable/disable
- **Stream responses**: On / Off
- **Code theme**: Select syntax highlighting
- **Markdown**: Enable/disable

### Notification Settings

- **Desktop notifications**: On / Off
- **Sound alerts**: On / Off
- **Email notifications**: Configure

## Keyboard Shortcuts

### General

| Shortcut | Action |
|----------|--------|
| `?` | Show help |
| `Ctrl + ,` | Open settings |
| `Ctrl + Shift + N` | New session |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl + 1` | Go to Chat |
| `Ctrl + 2` | Go to Sessions |
| `Ctrl + 3` | Go to Providers |
| `Ctrl + 4` | Go to Dashboard |

### Chat

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift + Enter` | New line |
| `Ctrl + K` | Clear chat |
| `Ctrl + ↑` | Previous message |
| `Ctrl + ↓` | Next message |
| `Ctrl + Shift + C` | Copy last response |

## Tips & Tricks

### Efficient Chat

1. **Be specific**: "Create a Python function to validate email addresses"
2. **Use examples**: "Like this: example@domain.com"
3. **Iterate**: Refine based on responses
4. **Use sessions**: Organize by project/topic

### Organizing Sessions

- Name sessions clearly: "Auth Module - JWT Implementation"
- Use tags: "backend", "security", "urgent"
- Set priorities for important tasks
- Archive old sessions regularly

### Monitoring Usage

- Check dashboard for spending trends
- Monitor provider performance
- Set up budget alerts
- Review session costs

## Troubleshooting

### Interface Won't Load

```bash
# Check backend
curl http://localhost:8000/api/health

# Check frontend
npm run dev  # In frontend directory
```

### Messages Not Sending

1. Check provider status (green dot?)
2. Verify network connection
3. Check browser console for errors
4. Try refreshing the page

### OODA Visualization Not Updating

1. Check WebSocket connection
2. Look for errors in browser console
3. Try reconnecting (refresh page)

## Mobile Access

The interface is responsive and works on mobile:
- Swipe to open sidebar
- Touch-optimized controls
- Collapsible panels

## Accessibility

- Keyboard navigation supported
- Screen reader compatible
- High contrast theme available
- Font size adjustable

## See Also

- [Quick Start](./quickstart.md)
- [Provider Setup](./providers.md)
- [API Documentation](../api/README.md)
