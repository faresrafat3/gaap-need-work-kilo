import { render, screen, act } from '@testing-library/react';
import { ToastProvider, useToast, Toast } from '@/components/common/Toast';

function TestComponent() {
  const toast = useToast();
  return (
    <button onClick={() => toast.success('Test toast', 'This is a test message')}>
      Show Toast
    </button>
  );
}

describe('Toast', () => {
  it('renders toast with correct content', () => {
    const onClose = jest.fn();
    render(
      <Toast
        id="test-1"
        type="success"
        title="Success!"
        message="Operation completed"
        onClose={onClose}
      />
    );
    expect(screen.getByText('Success!')).toBeInTheDocument();
    expect(screen.getByText('Operation completed')).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = jest.fn();
    render(
      <Toast
        id="test-1"
        type="error"
        title="Error!"
        onClose={onClose}
      />
    );
    const closeButton = screen.getByRole('button');
    closeButton.click();
    expect(onClose).toHaveBeenCalled();
  });

  it('applies correct variant styles', () => {
    const { container } = render(
      <Toast id="test-1" type="error" title="Error!" onClose={() => {}} />
    );
    expect(container.firstChild).toHaveClass('border-error/30');
  });
});

describe('ToastProvider', () => {
  it('provides toast context', () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );
    expect(screen.getByText('Show Toast')).toBeInTheDocument();
  });

  it('shows toast when triggered', async () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );
    
    const button = screen.getByText('Show Toast');
    await act(async () => {
      button.click();
    });
    
    expect(screen.getByText('Test toast')).toBeInTheDocument();
    expect(screen.getByText('This is a test message')).toBeInTheDocument();
  });
});