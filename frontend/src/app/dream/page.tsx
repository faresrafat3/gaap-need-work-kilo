'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { EmptyState } from '@/components/common/EmptyState';
import {
  Moon,
  Sparkles,
  ArrowRight,
  Brain,
  Database,
  Archive,
  Clock,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  Filter,
} from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface DreamLog {
  id: string;
  timestamp: string;
  duration_ms: number;
  memories_processed: number;
  lessons_extracted: number;
  tier_movements: TierMovement[];
  status: 'completed' | 'partial' | 'failed';
}

interface TierMovement {
  memory_id: string;
  memory_summary: string;
  from_tier: 'working' | 'short_term' | 'long_term';
  to_tier: 'working' | 'short_term' | 'long_term';
  reason: string;
}

interface Lesson {
  id: string;
  content: string;
  source_dream_id: string;
  created_at: string;
  applied_count: number;
  category: string;
}

interface DreamStats {
  total_dreams: number;
  total_lessons: number;
  avg_duration_ms: number;
  last_dream: string | null;
  memories_consolidated: number;
}

const dreamApi = {
  getLogs: (limit?: number) => api.get('/api/dream/logs', { params: { limit } }),
  getLessons: (limit?: number) => api.get('/api/dream/lessons', { params: { limit } }),
  getStats: () => api.get('/api/dream/stats'),
  trigger: () => api.post('/api/dream/trigger'),
};

const tierColors = {
  working: 'text-success border-success/30',
  short_term: 'text-layer3 border-layer3/30',
  long_term: 'text-warning border-warning/30',
};

const tierIcons = {
  working: Brain,
  short_term: Database,
  long_term: Archive,
};

function TimelineItem({ dream, isLast }: { dream: DreamLog; isLast: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const statusColors = {
    completed: 'bg-success',
    partial: 'bg-warning',
    failed: 'bg-error',
  };

  return (
    <div className="relative">
      <div className="absolute left-4 top-8 bottom-0 w-px bg-layer1/20" hidden={isLast} />
      <div className={`absolute left-2.5 top-2 w-3 h-3 rounded-full ${statusColors[dream.status]}`} />
      <div className="ml-10 pb-6">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full text-left group"
        >
          <div className="flex items-center gap-2 mb-1">
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )}
            <span className="font-medium group-hover:text-layer1 transition-colors">
              Dream #{dream.id.slice(0, 8)}
            </span>
            <Badge variant={dream.status === 'completed' ? 'success' : dream.status === 'partial' ? 'warning' : 'error'}>
              {dream.status}
            </Badge>
            <span className="text-sm text-gray-500 ml-auto">
              {formatDistanceToNow(new Date(dream.timestamp), { addSuffix: true })}
            </span>
          </div>
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="mt-3 space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 bg-cyber-dark rounded-lg">
                    <div className="text-xs text-gray-400">Duration</div>
                    <div className="font-mono">{(dream.duration_ms / 1000).toFixed(1)}s</div>
                  </div>
                  <div className="p-3 bg-cyber-dark rounded-lg">
                    <div className="text-xs text-gray-400">Memories</div>
                    <div className="font-mono">{dream.memories_processed}</div>
                  </div>
                  <div className="p-3 bg-cyber-dark rounded-lg">
                    <div className="text-xs text-gray-400">Lessons</div>
                    <div className="font-mono">{dream.lessons_extracted}</div>
                  </div>
                </div>

                {dream.tier_movements.length > 0 && (
                  <div>
                    <div className="text-sm text-gray-400 mb-2">Tier Movements</div>
                    <div className="space-y-2">
                      {dream.tier_movements.map((movement, i) => {
                        const FromIcon = tierIcons[movement.from_tier];
                        const ToIcon = tierIcons[movement.to_tier];
                        return (
                          <div
                            key={i}
                            className="flex items-center gap-3 p-2 bg-cyber-dark rounded-lg"
                          >
                            <div className={`flex items-center gap-1 ${tierColors[movement.from_tier]}`}>
                              <FromIcon className="w-3 h-3" />
                              <span className="text-xs capitalize">{movement.from_tier.replace('_', ' ')}</span>
                            </div>
                            <ArrowRight className="w-4 h-4 text-gray-500" />
                            <div className={`flex items-center gap-1 ${tierColors[movement.to_tier]}`}>
                              <ToIcon className="w-3 h-3" />
                              <span className="text-xs capitalize">{movement.to_tier.replace('_', ' ')}</span>
                            </div>
                            <div className="flex-1 text-xs text-gray-400 truncate">
                              {movement.reason}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function LessonCard({ lesson }: { lesson: Lesson }) {
  return (
    <div className="p-4 bg-cyber-dark rounded-lg border border-layer1/20">
      <div className="flex items-start justify-between mb-2">
        <Badge variant="default">{lesson.category}</Badge>
        <span className="text-xs text-gray-500">
          Applied {lesson.applied_count}x
        </span>
      </div>
      <p className="text-sm mb-2">{lesson.content}</p>
      <div className="text-xs text-gray-500">
        {formatDistanceToNow(new Date(lesson.created_at), { addSuffix: true })}
      </div>
    </div>
  );
}

export default function DreamPage() {
  const [filter, setFilter] = useState<'all' | 'completed' | 'partial' | 'failed'>('all');

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dream-stats'],
    queryFn: () => dreamApi.getStats(),
    refetchInterval: 30000,
  });

  const { data: logs, isLoading: logsLoading } = useQuery({
    queryKey: ['dream-logs', filter],
    queryFn: () => dreamApi.getLogs(50),
  });

  const { data: lessons } = useQuery({
    queryKey: ['dream-lessons'],
    queryFn: () => dreamApi.getLessons(20),
  });

  const filteredLogs = logs?.data?.filter((dream: DreamLog) =>
    filter === 'all' ? true : dream.status === filter
  );

  if (statsLoading || logsLoading) {
    return (
      <div className="flex h-screen bg-cyber-dark">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin w-8 h-8 border-2 border-layer1 border-t-transparent rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Dream" subtitle="Memory consolidation & learning" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
            <Card>
              <CardBody className="text-center">
                <Moon className="w-6 h-6 mx-auto mb-2 text-layer1" />
                <div className="text-2xl font-bold">{stats?.data?.total_dreams || 0}</div>
                <div className="text-sm text-gray-400">Total Dreams</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <Sparkles className="w-6 h-6 mx-auto mb-2 text-warning" />
                <div className="text-2xl font-bold">{stats?.data?.total_lessons || 0}</div>
                <div className="text-sm text-gray-400">Lessons Learned</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <Clock className="w-6 h-6 mx-auto mb-2 text-layer3" />
                <div className="text-2xl font-bold">
                  {((stats?.data?.avg_duration_ms || 0) / 1000).toFixed(1)}s
                </div>
                <div className="text-sm text-gray-400">Avg Duration</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <Database className="w-6 h-6 mx-auto mb-2 text-success" />
                <div className="text-2xl font-bold">{stats?.data?.memories_consolidated || 0}</div>
                <div className="text-sm text-gray-400">Memories Moved</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <RefreshCw className="w-6 h-6 mx-auto mb-2 text-gray-400" />
                <div className="text-sm font-medium truncate">
                  {stats?.data?.last_dream
                    ? formatDistanceToNow(new Date(stats.data.last_dream), { addSuffix: true })
                    : 'Never'}
                </div>
                <div className="text-sm text-gray-400">Last Dream</div>
              </CardBody>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <h3 className="font-semibold">Dream Timeline</h3>
                    <div className="flex items-center gap-1">
                      <Filter className="w-4 h-4 text-gray-400" />
                      <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value as typeof filter)}
                        className="bg-cyber-dark border border-layer1/30 rounded px-2 py-1 text-sm"
                      >
                        <option value="all">All</option>
                        <option value="completed">Completed</option>
                        <option value="partial">Partial</option>
                        <option value="failed">Failed</option>
                      </select>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" leftIcon={<Moon className="w-4 h-4" />}>
                    Trigger Dream
                  </Button>
                </div>
              </CardHeader>
              <CardBody>
                {filteredLogs?.length > 0 ? (
                  <div className="relative">
                    {filteredLogs.map((dream: DreamLog, i: number) => (
                      <TimelineItem
                        key={dream.id}
                        dream={dream}
                        isLast={i === filteredLogs.length - 1}
                      />
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    icon={<Moon className="w-12 h-12 text-gray-600" />}
                    title="No dreams yet"
                    description="Dreams will appear here after consolidation runs"
                  />
                )}
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">Lessons Extracted</h3>
              </CardHeader>
              <CardBody>
                {lessons?.data?.length > 0 ? (
                  <div className="space-y-3 max-h-[500px] overflow-y-auto">
                    {lessons.data.map((lesson: Lesson) => (
                      <LessonCard key={lesson.id} lesson={lesson} />
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No lessons yet"
                    description="Lessons will appear after dream consolidation"
                  />
                )}
              </CardBody>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}