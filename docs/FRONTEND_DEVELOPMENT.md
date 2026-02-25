# GAAP Frontend Developer Guide

## Overview

The GAAP Frontend is a modern, responsive web application built with Next.js 14 using the App Router architecture. It provides a comprehensive control interface for the General AI Agent Platform, featuring real-time updates, interactive visualizations, and a distinctive Cyber-Noir aesthetic.

### Architecture Highlights

- **Server Components** for initial page loads and SEO
- **Client Components** for interactive features
- **Real-time Updates** via WebSocket connections
- **Optimistic UI** with React Query for data fetching
- **Global State** managed by Zustand stores
- **Accessible** with screen reader support and keyboard navigation

---

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 14.1.0 | React framework with App Router |
| React | 18.2.0 | UI library |
| TypeScript | 5.3.0 | Type safety |
| Tailwind CSS | 3.4.0 | Styling |
| Zustand | 4.5.0 | State management |
| React Query | 5.17.0 | Server state management |
| XYFlow | 12.0.0 | Graph visualization (OODA flow) |
| Recharts | 2.10.0 | Charts and data visualization |
| Framer Motion | 11.0.0 | Animations |
| Axios | 1.6.0 | HTTP client |
| Lucide React | 0.314.0 | Icons |
| date-fns | 3.2.0 | Date utilities |
| clsx | 2.1.0 | Conditional classNames |

### Development Dependencies

- **Jest** + **React Testing Library** - Unit testing
- **Playwright** - E2E testing
- **ESLint** - Linting
- **Prettier** - Code formatting
- **MSW** - API mocking in tests

---

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard (/)
│   │   ├── providers.tsx       # React Query, Toast providers
│   │   ├── globals.css         # Global styles
│   │   ├── config/             # Configuration page
│   │   ├── providers/          # Provider management
│   │   ├── research/           # Research interface
│   │   ├── sessions/           # Session management
│   │   ├── healing/            # Self-healing monitor
│   │   ├── memory/             # Memory dashboard
│   │   ├── budget/             # Budget tracking
│   │   ├── security/           # Security status
│   │   └── debt/               # Technical debt tracker
│   │
│   ├── components/
│   │   ├── common/             # 17 reusable components
│   │   ├── dashboard/          # Dashboard widgets
│   │   ├── layout/             # Header, Sidebar
│   │   ├── ooda/               # OODA loop visualization
│   │   ├── steering/           # Agent steering controls
│   │   └── export/             # Export functionality
│   │
│   ├── stores/                 # Zustand stores (5 stores)
│   │   ├── configStore.ts
│   │   ├── providerStore.ts
│   │   ├── sessionStore.ts
│   │   ├── uiStore.ts
│   │   └── eventStore.ts
│   │
│   ├── hooks/                  # Custom hooks
│   │   ├── useWebSocket.ts
│   │   ├── useEvents.ts
│   │   └── useKeyboardShortcuts.ts
│   │
│   ├── lib/
│   │   ├── api.ts              # API client
│   │   ├── types.ts            # TypeScript types
│   │   ├── utils.ts            # Utility functions
│   │   └── accessibility.ts    # A11y helpers
│   │
│   └── __tests__/              # Test files
│
├── e2e/                        # Playwright E2E tests
├── jest.config.ts              # Jest configuration
├── playwright.config.ts        # Playwright configuration
├── tailwind.config.js          # Tailwind configuration
├── next.config.js              # Next.js configuration
└── tsconfig.json               # TypeScript configuration
```

---

## Common Components

### Button

The primary interactive element with multiple variants and sizes.

```tsx
import { Button } from '@/components/common';

// Variants: primary, secondary, success, warning, error, ghost, outline
// Sizes: xs, sm, md, lg, xl

<Button variant="primary" size="md">Click me</Button>
<Button variant="error" isLoading>Deleting...</Button>
<Button variant="outline" leftIcon={<Icon />}>With Icon</Button>
<Button isFullWidth>Full Width</Button>
```

**Props:**
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| variant | `ButtonVariant` | `'primary'` | Visual style |
| size | `ButtonSize` | `'md'` | Button size |
| isLoading | `boolean` | `false` | Shows spinner, disables button |
| isFullWidth | `boolean` | `false` | Full width button |
| leftIcon | `ReactNode` | - | Icon before text |
| rightIcon | `ReactNode` | - | Icon after text |

---

### Card

Container component for grouped content.

```tsx
import { Card, CardHeader, CardBody, CardFooter } from '@/components/common';

// Variants: default, bordered, elevated, interactive
// Padding: none, sm, md, lg

<Card variant="elevated" padding="lg">
  <CardHeader>Title</CardHeader>
  <CardBody>Content here</CardBody>
  <CardFooter>Actions</CardFooter>
</Card>
```

---

### Input Components

Form inputs with labels, error states, and hints.

```tsx
import { Input, Textarea, Checkbox, Switch } from '@/components/common';

// Input with label and error
<Input
  label="API Key"
  placeholder="Enter your key"
  error="Key is required"
  leftElement={<KeyIcon />}
/>

// Textarea
<Textarea
  label="Description"
  placeholder="Enter description"
  rows={4}
/>

// Checkbox
<Checkbox label="Remember me" checked={remember} onChange={...} />

// Switch (toggle)
<Switch
  label="Enable feature"
  description="This will enable the feature"
  checked={enabled}
  onChange={...}
/>
```

---

### Select

Dropdown select component.

```tsx
import { Select } from '@/components/common';

<Select
  label="Provider"
  options={[
    { value: 'openai', label: 'OpenAI' },
    { value: 'anthropic', label: 'Anthropic', disabled: true },
    { value: 'local', label: 'Local Model' },
  ]}
  placeholder="Select a provider"
  error="Provider is required"
/>
```

---

### Table

Data table with sorting and row actions.

```tsx
import { Table } from '@/components/common';

const columns = [
  { key: 'name', header: 'Name' },
  { key: 'status', header: 'Status', render: (value) => <Badge>{value}</Badge> },
  { key: 'actions', header: '', align: 'right', render: (_, row) => <Actions row={row} /> },
];

<Table
  columns={columns}
  data={items}
  keyExtractor={(row) => row.id}
  isLoading={loading}
  emptyMessage="No items found"
  onRowClick={(row) => handleSelect(row)}
/>
```

---

### Modal

Portal-based modal with focus trap and keyboard navigation.

```tsx
import { Modal, ModalHeader, ModalBody, ModalFooter } from '@/components/common';
import { Button } from '@/components/common';

const [isOpen, setIsOpen] = useState(false);

<Modal isOpen={isOpen} onClose={() => setIsOpen(false)} size="lg">
  <ModalHeader>Confirm Action</ModalHeader>
  <ModalBody>
    Are you sure you want to proceed?
  </ModalBody>
  <ModalFooter>
    <Button variant="ghost" onClick={() => setIsOpen(false)}>Cancel</Button>
    <Button variant="error" onClick={handleConfirm}>Confirm</Button>
  </ModalFooter>
</Modal>
```

**Sizes:** `sm`, `md`, `lg`, `xl`, `full`

---

### Toast

Notification system using React Context.

```tsx
import { useToast } from '@/components/common';

function MyComponent() {
  const toast = useToast();
  
  const handleSave = async () => {
    try {
      await saveData();
      toast.success('Saved successfully');
    } catch (error) {
      toast.error('Save failed', error.message);
    }
  };
  
  return <Button onClick={handleSave}>Save</Button>;
}

// Types: success, error, warning, info
toast.success('Title', 'Optional message', 5000); // duration in ms, 0 for persistent
```

---

### Loading Components

```tsx
import { Loading, Spinner, Skeleton, TableSkeleton, CardSkeleton } from '@/components/common';

// Full page loading
<Loading fullScreen text="Loading..." />

// Inline spinner
<Spinner size="sm" />

// Skeleton loading
<Skeleton variant="text" width="100%" height={20} />
<Skeleton variant="circular" width={40} height={40} />

// Pre-built skeletons
<TableSkeleton rows={5} columns={4} />
<CardSkeleton />
```

---

### ErrorBoundary

Error boundary for graceful error handling.

```tsx
import { ErrorBoundary, useErrorBoundary } from '@/components/common';

// Class component boundary
<ErrorBoundary onError={(error, info) => logError(error)}>
  <RiskyComponent />
</ErrorBoundary>

// Hook for functional components
function MyComponent() {
  const { showBoundary } = useErrorBoundary();
  
  const handleClick = () => {
    try {
      riskyOperation();
    } catch (error) {
      showBoundary(error);
    }
  };
}
```

---

### Badge

Status indicators and labels.

```tsx
import { Badge } from '@/components/common';

// Variants: default, success, warning, error, info, layer1, layer2, layer3
// Sizes: sm, md, lg

<Badge variant="success">Active</Badge>
<Badge variant="warning" dot>Warning</Badge>
<Badge variant="error" size="sm">Error</Badge>
```

---

### Progress

Linear and circular progress indicators.

```tsx
import { Progress, CircularProgress } from '@/components/common';

// Linear progress
<Progress value={75} max={100} variant="success" showLabel />

// Circular progress
<CircularProgress value={75} size={80} variant="default" />
```

---

### EmptyState

Placeholder for empty data states.

```tsx
import { EmptyState } from '@/components/common';
import { Button } from '@/components/common';

<EmptyState
  icon={<InboxIcon />}
  title="No sessions found"
  description="Create your first session to get started"
  action={<Button>Create Session</Button>}
/>
```

---

### Search (Command Palette)

Global search with keyboard shortcuts (Cmd+K).

```tsx
import { SearchCommand } from '@/components/common';

// Default search with navigation items
<SearchCommand />

// Custom items
<SearchCommand
  items={[
    { id: '1', title: 'Dashboard', href: '/', category: 'Navigation' },
    { id: '2', title: 'Settings', href: '/settings', category: 'Admin' },
  ]}
  placeholder="Search..."
/>
```

---

### Accessibility Components

```tsx
import { SkipToContent, AnnouncerProvider, useAnnouncer } from '@/components/common';

// Skip to content link (in layout)
<SkipToContent />

// Screen reader announcements
function MyComponent() {
  const announce = useAnnouncer();
  
  const handleAction = () => {
    doSomething();
    announce('Action completed successfully');
  };
}
```

---

### KeyboardShortcutsHelp

Display available keyboard shortcuts.

```tsx
import { KeyboardShortcutsHelp } from '@/components/common';

<KeyboardShortcutsHelp />
```

---

## State Management

Zustand stores provide global state management with optional persistence.

### configStore

Manages application configuration state.

```tsx
import { useConfigStore } from '@/stores/configStore';

function ConfigEditor() {
  const {
    config,
    selectedModule,
    hasUnsavedChanges,
    setConfig,
    updateModule,
    setSelectedModule,
  } = useConfigStore();
  
  // Update a module's configuration
  updateModule('providers', { openai: { enabled: true } });
  
  // Check for unsaved changes
  if (hasUnsavedChanges) {
    // Prompt user to save
  }
}
```

**State:**
- `config` - Current configuration object
- `selectedModule` - Currently selected module
- `hasUnsavedChanges` - Dirty state flag
- `isLoading` - Loading state
- `error` - Error message

**Persistence:** Session storage (selectedModule only)

---

### providerStore

Manages LLM provider state.

```tsx
import { useProviderStore } from '@/stores/providerStore';

function ProviderList() {
  const {
    providers,
    selectedProvider,
    setProviders,
    updateProvider,
    getHealthyProviders,
  } = useProviderStore();
  
  const healthyProviders = getHealthyProviders();
}
```

**Derived Selectors:**
- `getProvider(name)` - Get provider by name
- `getEnabledProviders()` - Get all enabled providers
- `getHealthyProviders()` - Get healthy, enabled providers

---

### sessionStore

Manages active sessions.

```tsx
import { useSessionStore } from '@/stores/sessionStore';

function SessionsPage() {
  const {
    sessions,
    activeSessionId,
    addSession,
    updateSession,
    setActiveSession,
    getActiveSession,
    getRunningSessions,
  } = useSessionStore();
}
```

**Persistence:** Session storage (activeSessionId only)

---

### uiStore

Manages UI state (theme, sidebar, modals).

```tsx
import { useUIStore } from '@/stores/uiStore';

function Layout() {
  const {
    theme,
    sidebarState,
    activeModal,
    searchOpen,
    toggleSidebar,
    openModal,
    closeModal,
    setSearchOpen,
  } = useUIStore();
}
```

**Persistence:** Local storage (theme, sidebarState, shortcutsEnabled)

---

### eventStore

Manages real-time events from WebSocket.

```tsx
import { useEventStore } from '@/stores/eventStore';

function EventsList() {
  const {
    events,
    isConnected,
    eventTypeFilter,
    addEvent,
    setEventTypeFilter,
    getFilteredEvents,
  } = useEventStore();
}
```

---

## Hooks

### useWebSocket

Manages WebSocket connection to backend.

```tsx
import { useWebSocket } from '@/hooks/useWebSocket';

function MyComponent() {
  const {
    isConnected,
    lastEvent,
    events,
    send,
    pause,
    resume,
    veto,
  } = useWebSocket('events');
  
  // Send a message
  send({ type: 'custom', data: {} });
  
  // Steering commands
  pause(sessionId);
  resume(sessionId, 'Optional instruction');
  veto(sessionId);
}
```

**Features:**
- Auto-reconnect on disconnect (3s delay)
- Keep-alive ping (30s interval)
- Event buffering (max 100 events)
- Channel-based subscriptions

---

### useEvents

Simplified hook for event consumption.

```tsx
import { useEvents } from '@/hooks/useEvents';

function EventsList() {
  const { connectionStatus, lastEvent, events } = useEvents();
}
```

---

### useKeyboardShortcuts

Register keyboard shortcuts.

```tsx
import { useKeyboardShortcuts, useAppShortcuts } from '@/hooks/useKeyboardShortcuts';

// Custom shortcuts
function MyComponent() {
  useKeyboardShortcuts([
    { key: 's', ctrl: true, action: () => save(), description: 'Save' },
    { key: 'Escape', action: () => close(), description: 'Close' },
  ]);
}

// App-wide shortcuts (already registered in layout)
function App() {
  useAppShortcuts(); // Registers: Ctrl+K (search), Ctrl+B (sidebar), etc.
}
```

**Default Shortcuts:**
| Shortcut | Action |
|----------|--------|
| `Ctrl+K` / `/` | Open search |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+R` | Refresh page |
| `?` | Open help |
| `Escape` | Close modal |
| `Ctrl+1-5` | Navigate pages |

---

## API Integration

### REST API Client

The API client is organized by domain with axios.

```tsx
import {
  configApi,
  providersApi,
  sessionsApi,
  healingApi,
  memoryApi,
  budgetApi,
  systemApi,
  researchApi,
  securityApi,
  debtApi,
} from '@/lib/api';

// Configuration
const config = await configApi.get();
await configApi.updateModule('providers', { openai: { enabled: true } });

// Providers
const providers = await providersApi.list();
await providersApi.test('openai');
await providersApi.enable('openai');

// Sessions
const sessions = await sessionsApi.list();
const session = await sessionsApi.create({ type: 'research', name: 'My Session' });
await sessionsApi.pause(sessionId);

// Healing
const history = await healingApi.getHistory(50);
const patterns = await healingApi.getPatterns();

// Memory
const stats = await memoryApi.getStats();
const results = await memoryApi.search('query');

// Budget
const budget = await budgetApi.get();
const alerts = await budgetApi.getAlerts();

// System
const health = await systemApi.getHealth();
const metrics = await systemApi.getMetrics();

// Research
const results = await researchApi.search('query', 2); // depth=2

// Security
const status = await securityApi.getStatus();
await securityApi.scan();

// Debt
const summary = await debtApi.getSummary();
await debtApi.scan();
```

### React Query Integration

Use React Query for data fetching with caching.

```tsx
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { providersApi } from '@/lib/api';

function ProvidersList() {
  const queryClient = useQueryClient();
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['providers'],
    queryFn: () => providersApi.list(),
  });
  
  const testMutation = useMutation({
    mutationFn: (name: string) => providersApi.test(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
    },
  });
  
  return (
    <div>
      {data?.map(provider => (
        <Button onClick={() => testMutation.mutate(provider.name)}>
          Test
        </Button>
      ))}
    </div>
  );
}
```

---

## Styling

### Tailwind CSS

The project uses Tailwind CSS with a custom Cyber-Noir theme.

**Theme Colors (`tailwind.config.js`):**

```javascript
colors: {
  layer0: '#1a1a2e',
  layer1: '#4a0e78',    // Primary purple
  layer2: '#0d47a1',    // Blue
  layer3: '#2e7d32',    // Green
  healing: '#e65100',
  error: '#b71c1c',
  warning: '#f57c00',
  success: '#00c853',
  cyber: {
    dark: '#0a0a0f',
    darker: '#050508',
    primary: '#00ff88',
    secondary: '#00ccff',
    accent: '#ff00ff',
  }
}
```

**Custom Animations:**

```css
.animate-pulse-slow  /* Slow pulse animation */
.animate-glow        /* Glow effect */
```

### Dark Mode Only

The application is dark mode only. The `<html>` element has `class="dark"` set permanently.

### Common Patterns

```tsx
// Card background
<div className="bg-cyber-darker border border-layer1/30 rounded-lg">

// Text colors
<span className="text-cyber-primary">Primary accent</span>
<span className="text-gray-400">Secondary text</span>

// Interactive element
<button className="hover:bg-layer1/10 hover:text-white transition-all">

// Glow effect
<div className="glow-layer1">

// Status indicators
<span className="text-success">Healthy</span>
<span className="text-warning">Degraded</span>
<span className="text-error">Error</span>
```

---

## Testing

### Unit Tests (Jest)

```bash
npm test              # Run all tests
npm run test:watch    # Watch mode
npm run test:coverage # Coverage report
```

**Test File Location:** `src/__tests__/`

**Example Test:**

```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '@/components/common/Button';

describe('Button', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Click me');
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('shows loading state', () => {
    render(<Button isLoading>Click me</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### E2E Tests (Playwright)

```bash
npm run test:e2e      # Run E2E tests
```

**Test File Location:** `e2e/`

**Playwright Configuration:**
- Runs on Chromium, Firefox, and WebKit
- Auto-starts dev server
- Traces on retry

---

## Development Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server (port 3000) |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm test` | Run unit tests |
| `npm run test:watch` | Run tests in watch mode |
| `npm run test:coverage` | Run tests with coverage |
| `npm run test:e2e` | Run E2E tests |
| `npm run type-check` | TypeScript type check |
| `npm run lint` | Run ESLint |
| `npm run format` | Format with Prettier |

---

## Adding New Features

### Creating a New Page

1. Create the page directory and file:

```bash
mkdir -p src/app/new-feature
touch src/app/new-feature/page.tsx
```

2. Create the page component:

```tsx
// src/app/new-feature/page.tsx
'use client';

import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common';

export default function NewFeaturePage() {
  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="New Feature" />
        <main className="flex-1 overflow-y-auto p-6">
          <Card>
            <CardHeader>New Feature</CardHeader>
            <CardBody>
              Content here
            </CardBody>
          </Card>
        </main>
      </div>
    </div>
  );
}
```

3. Add navigation item in `src/components/layout/Sidebar.tsx`:

```tsx
const navItems = [
  // ... existing items
  { name: 'New Feature', href: '/new-feature', icon: NewIcon },
];
```

---

### Creating a New Component

1. Create the component file:

```tsx
// src/components/common/MyComponent.tsx
'use client';

import { forwardRef, HTMLAttributes } from 'react';
import { clsx } from 'clsx';

export interface MyComponentProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'special';
}

export const MyComponent = forwardRef<HTMLDivElement, MyComponentProps>(
  ({ className, variant = 'default', children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx(
          'base-classes-here',
          variant === 'special' && 'special-classes',
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

MyComponent.displayName = 'MyComponent';
```

2. Export from `src/components/common/index.ts`:

```tsx
export { MyComponent } from './MyComponent';
export type { MyComponentProps } from './MyComponent';
```

3. Create tests:

```tsx
// src/__tests__/components/common/MyComponent.test.tsx
import { render, screen } from '@testing-library/react';
import { MyComponent } from '@/components/common/MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent>Test</MyComponent>);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

---

### Adding a New Store

1. Create the store file:

```tsx
// src/stores/myStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface MyState {
  items: string[];
  selectedItem: string | null;
  
  setItems: (items: string[]) => void;
  addItem: (item: string) => void;
  removeItem: (item: string) => void;
  setSelectedItem: (item: string | null) => void;
}

export const useMyStore = create<MyState>()(
  persist(
    (set, get) => ({
      items: [],
      selectedItem: null,
      
      setItems: (items) => set({ items }),
      
      addItem: (item) =>
        set((state) => ({
          items: [...state.items, item],
        })),
      
      removeItem: (item) =>
        set((state) => ({
          items: state.items.filter((i) => i !== item),
          selectedItem: state.selectedItem === item ? null : state.selectedItem,
        })),
      
      setSelectedItem: (item) => set({ selectedItem: item }),
    }),
    {
      name: 'gaap-my-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        selectedItem: state.selectedItem,
      }),
    }
  )
);
```

2. Export from `src/stores/index.ts`:

```tsx
export { useMyStore } from './myStore';
```

---

### Adding a New Hook

```tsx
// src/hooks/useMyHook.ts
'use client';

import { useState, useEffect, useCallback } from 'react';

export function useMyHook(initialValue: string) {
  const [value, setValue] = useState(initialValue);
  const [isLoading, setIsLoading] = useState(false);
  
  const updateValue = useCallback((newValue: string) => {
    setIsLoading(true);
    // Do something
    setValue(newValue);
    setIsLoading(false);
  }, []);
  
  useEffect(() => {
    // Setup
    return () => {
      // Cleanup
    };
  }, []);
  
  return {
    value,
    isLoading,
    updateValue,
  };
}
```

---

## Best Practices

### Component Structure

```tsx
// 1. 'use client' directive if needed
'use client';

// 2. Imports (grouped)
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { clsx } from 'clsx';

import { Button, Card } from '@/components/common';
import { useUIStore } from '@/stores/uiStore';
import { providersApi } from '@/lib/api';

// 3. Types
interface MyComponentProps {
  title: string;
  onAction?: () => void;
}

// 4. Component
export function MyComponent({ title, onAction }: MyComponentProps) {
  // Hooks
  const router = useRouter();
  const { theme } = useUIStore();
  const [isOpen, setIsOpen] = useState(false);
  
  // Effects
  useEffect(() => {
    // ...
  }, []);
  
  // Handlers
  const handleClick = () => {
    onAction?.();
    setIsOpen(false);
  };
  
  // Render
  return (
    <Card>
      <h2>{title}</h2>
      <Button onClick={handleClick}>Action</Button>
    </Card>
  );
}
```

### Type Safety

- Always define interfaces for props
- Use TypeScript discriminated unions for variants
- Export types alongside components
- Use `unknown` instead of `any` for API responses

```tsx
// Good
interface ButtonProps {
  variant?: 'primary' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
}

// Bad
interface ButtonProps {
  variant?: string;
  size?: string;
}
```

### Accessibility

1. **Semantic HTML:** Use proper heading hierarchy, landmarks, and lists
2. **ARIA Labels:** Add labels to interactive elements
3. **Keyboard Navigation:** All interactive elements focusable and operable via keyboard
4. **Focus Management:** Trap focus in modals, restore focus on close
5. **Announcements:** Use `useAnnouncer` for screen reader updates

```tsx
// Good
<button
  onClick={handleClick}
  aria-label="Close dialog"
  aria-pressed={isActive}
>
  <XIcon />
</button>

// Modal focus trap
<Modal
  isOpen={isOpen}
  onClose={handleClose}
>
  {/* Focus automatically trapped */}
</Modal>
```

### Error Handling

```tsx
// API calls
const fetchData = async () => {
  try {
    const response = await api.get('/data');
    setData(response.data);
  } catch (error) {
    if (axios.isAxiosError(error)) {
      toast.error('Failed to fetch data', error.message);
    } else {
      toast.error('An unexpected error occurred');
    }
  }
};

// Error boundary
<ErrorBoundary onError={(error) => logToService(error)}>
  <RiskyComponent />
</ErrorBoundary>
```

### Performance

1. **Memoization:** Use `useMemo` and `useCallback` for expensive operations
2. **Code Splitting:** Use dynamic imports for large components
3. **React Query:** Leverage caching and background refetching
4. **Virtualization:** For long lists, use virtualization

```tsx
// Dynamic import
const HeavyComponent = dynamic(
  () => import('./HeavyComponent'),
  { loading: () => <Loading /> }
);

// Memoization
const sortedItems = useMemo(
  () => items.sort((a, b) => a.name.localeCompare(b.name)),
  [items]
);

const handleSelect = useCallback(
  (id: string) => {
    setSelected(id);
    onSelect?.(id);
  },
  [onSelect]
);
```

---

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## Troubleshooting

### Common Issues

**Hydration Error:**
- Ensure client components have consistent initial state
- Use `suppressHydrationWarning` on elements with browser-specific values

**WebSocket Connection Issues:**
- Check `NEXT_PUBLIC_WS_URL` is correct
- Verify backend WebSocket endpoint is running
- Check browser console for connection errors

**Build Errors:**
- Run `npm run type-check` to find TypeScript errors
- Run `npm run lint` to find linting errors
- Clear `.next` cache: `rm -rf .next`

**Test Failures:**
- Clear Jest cache: `npm test -- --clearCache`
- Check for missing mocks in `jest.setup.ts`

---

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Zustand Documentation](https://zustand-demo.pmnd.rs/)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Framer Motion Documentation](https://www.framer.com/motion/)
