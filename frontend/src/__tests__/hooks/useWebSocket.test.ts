import { renderHook, act } from '@testing-library/react';
import { useWebSocket } from '@/hooks/useWebSocket';

describe('useWebSocket', () => {
  it('initializes with disconnected state', () => {
    const { result } = renderHook(() => useWebSocket('events'));
    expect(result.current.isConnected).toBe(false);
  });

  it('provides send function', () => {
    const { result } = renderHook(() => useWebSocket('events'));
    expect(result.current.send).toBeDefined();
    expect(typeof result.current.send).toBe('function');
  });

  it('provides steering functions', () => {
    const { result } = renderHook(() => useWebSocket('steering'));
    expect(result.current.pause).toBeDefined();
    expect(result.current.resume).toBeDefined();
    expect(result.current.veto).toBeDefined();
  });
});