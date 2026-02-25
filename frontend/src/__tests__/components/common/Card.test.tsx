import { render, screen } from '@testing-library/react';
import { Card, CardHeader, CardBody, CardFooter } from '@/components/common/Card';

describe('Card', () => {
  it('renders correctly', () => {
    render(<Card>Card content</Card>);
    expect(screen.getByText('Card content')).toBeInTheDocument();
  });

  it('applies variant styles', () => {
    const { container } = render(<Card variant="bordered">Bordered</Card>);
    expect(container.firstChild).toHaveClass('border-2');
  });

  it('applies padding styles', () => {
    const { container, rerender } = render(<Card padding="none">No padding</Card>);
    expect(container.firstChild).not.toHaveClass('p-3');
    expect(container.firstChild).not.toHaveClass('p-4');
    expect(container.firstChild).not.toHaveClass('p-6');

    rerender(<Card padding="lg">Large padding</Card>);
    expect(container.firstChild).toHaveClass('p-6');
  });
});

describe('CardHeader', () => {
  it('renders correctly', () => {
    render(
      <Card>
        <CardHeader>Header</CardHeader>
      </Card>
    );
    expect(screen.getByText('Header')).toBeInTheDocument();
  });
});

describe('CardBody', () => {
  it('renders correctly', () => {
    render(
      <Card>
        <CardBody>Body content</CardBody>
      </Card>
    );
    expect(screen.getByText('Body content')).toBeInTheDocument();
  });
});

describe('CardFooter', () => {
  it('renders correctly', () => {
    render(
      <Card>
        <CardFooter>Footer</CardFooter>
      </Card>
    );
    expect(screen.getByText('Footer')).toBeInTheDocument();
  });
});