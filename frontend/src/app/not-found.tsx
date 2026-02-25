import Link from 'next/link';
import { Button } from '@/components/common/Button';
import { Home, Search } from 'lucide-react';

export default function NotFoundPage() {
  return (
    <div className="flex h-screen flex-col items-center justify-center bg-cyber-dark p-4">
      <h1 className="text-6xl font-bold text-layer1 mb-4">404</h1>
      <h2 className="text-xl font-semibold mb-2">Page Not Found</h2>
      <p className="text-gray-400 mb-6 text-center">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <div className="flex gap-3">
        <Link href="/">
          <Button leftIcon={<Home className="w-4 h-4" />}>Go Home</Button>
        </Link>
        <Link href="/search">
          <Button variant="secondary" leftIcon={<Search className="w-4 h-4" />}>
            Search
          </Button>
        </Link>
      </div>
    </div>
  );
}
