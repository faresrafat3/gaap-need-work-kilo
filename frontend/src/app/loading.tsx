import { Loading } from '@/components/common/Loading';

export default function LoadingPage() {
  return (
    <div className="flex h-screen items-center justify-center bg-cyber-dark">
      <Loading size="lg" text="Loading..." />
    </div>
  );
}
