'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { ToastProvider } from '@/components/common/Toast';
import { AnnouncerProvider } from '@/components/common/Announcer';
import { SkipToContent } from '@/components/common/SkipToContent';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <AnnouncerProvider>
        <ToastProvider>
          <SkipToContent />
          {children}
        </ToastProvider>
      </AnnouncerProvider>
    </QueryClientProvider>
  );
}
