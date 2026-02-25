'use client';

import { useEffect } from 'react';
import { Button } from '@/components/common/Button';
import { AlertTriangle } from 'lucide-react';

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex h-screen flex-col items-center justify-center bg-cyber-dark p-4">
      <AlertTriangle className="w-16 h-16 text-error mb-6" />
      <h1 className="text-2xl font-bold mb-2">Something went wrong!</h1>
      <p className="text-gray-400 mb-6 text-center max-w-md">
        {error.message || 'An unexpected error occurred'}
      </p>
      <Button onClick={reset}>Try again</Button>
    </div>
  );
}
