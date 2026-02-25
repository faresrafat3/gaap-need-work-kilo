'use client';

import { useState } from 'react';
import { clsx } from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, ChevronRight } from 'lucide-react';

export type LayerType = 'strategy' | 'tactics' | 'execution';

export interface LayerInfo {
  type: LayerType;
  name: string;
  description: string;
  status: 'idle' | 'active' | 'completed' | 'error';
  color: string;
  bgColor: string;
  borderColor: string;
}

const layerConfigs: Record<LayerType, LayerInfo> = {
  strategy: {
    type: 'strategy',
    name: 'Strategy',
    description: 'High-level planning and goal setting',
    status: 'idle',
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/20',
    borderColor: 'border-purple-500/30',
  },
  tactics: {
    type: 'tactics',
    name: 'Tactics',
    description: 'Mid-level approach and methods',
    status: 'idle',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/20',
    borderColor: 'border-blue-500/30',
  },
  execution: {
    type: 'execution',
    name: 'Execution',
    description: 'Low-level implementation details',
    status: 'idle',
    color: 'text-green-400',
    bgColor: 'bg-green-500/20',
    borderColor: 'border-green-500/30',
  },
};

const statusIcons: Record<string, string> = {
  idle: '○',
  active: '●',
  completed: '✓',
  error: '✕',
};

interface LayerNavigationProps {
  currentLayer?: LayerType;
  layers?: Partial<Record<LayerType, Partial<LayerInfo>>>;
  onLayerSelect?: (layer: LayerType) => void;
  showDescriptions?: boolean;
  compact?: boolean;
}

export function LayerNavigation({
  currentLayer = 'strategy',
  layers = {},
  onLayerSelect,
  showDescriptions = false,
  compact = false,
}: LayerNavigationProps) {
  const [hoveredLayer, setHoveredLayer] = useState<LayerType | null>(null);
  const [expandedLayer, setExpandedLayer] = useState<LayerType | null>(null);

  const getLayerInfo = (type: LayerType): LayerInfo => {
    const config = layerConfigs[type];
    const overrides = layers[type] || {};
    return { ...config, ...overrides } as LayerInfo;
  };

  const layerTypes: LayerType[] = ['strategy', 'tactics', 'execution'];

  if (compact) {
    return (
      <div className="flex items-center gap-1 p-1 bg-cyber-darker rounded-lg">
        {layerTypes.map((type) => {
          const info = getLayerInfo(type);
          const isActive = currentLayer === type;

          return (
            <button
              key={type}
              onClick={() => onLayerSelect?.(type)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all text-sm',
                isActive ? `${info.bgColor} ${info.color}` : 'text-gray-400 hover:bg-cyber-dark'
              )}
            >
              <span className={clsx('font-mono', isActive && info.color)}>
                {statusIcons[info.status]}
              </span>
              <span>{info.name}</span>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="bg-cyber-darker rounded-lg border border-layer1/20 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-layer1/20 text-gray-400">
        <Layers className="w-4 h-4" />
        <span className="text-sm font-medium">Layer Navigation</span>
      </div>

      <div className="p-2">
        <div className="relative flex flex-col gap-2">
          {layerTypes.map((type, index) => {
            const info = getLayerInfo(type);
            const isActive = currentLayer === type;
            const isHovered = hoveredLayer === type;
            const isExpanded = expandedLayer === type;

            return (
              <div key={type} className="relative">
                {index < layerTypes.length - 1 && (
                  <div
                    className={clsx(
                      'absolute left-5 top-10 w-px h-2',
                      currentLayer === type || currentLayer === layerTypes[index + 1]
                        ? info.borderColor
                        : 'bg-layer1/20'
                    )}
                  />
                )}

                <motion.button
                  onClick={() => {
                    if (showDescriptions) {
                      setExpandedLayer(isExpanded ? null : type);
                    }
                    onLayerSelect?.(type);
                  }}
                  onMouseEnter={() => setHoveredLayer(type)}
                  onMouseLeave={() => setHoveredLayer(null)}
                  className={clsx(
                    'w-full flex items-center gap-3 p-2 rounded-lg transition-all text-left',
                    isActive
                      ? `${info.bgColor} ${info.borderColor} border`
                      : 'hover:bg-cyber-dark',
                    isHovered && !isActive && 'bg-cyber-dark/50'
                  )}
                  whileHover={{ x: 2 }}
                >
                  <div
                    className={clsx(
                      'flex items-center justify-center w-8 h-8 rounded-lg',
                      info.bgColor
                    )}
                  >
                    <span className={clsx('text-lg font-mono', info.color)}>
                      {statusIcons[info.status]}
                    </span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className={clsx('font-medium', isActive ? info.color : 'text-white')}>
                        {info.name}
                      </span>
                      {showDescriptions && (
                        <ChevronRight
                          className={clsx(
                            'w-4 h-4 text-gray-400 transition-transform',
                            isExpanded && 'rotate-90'
                          )}
                        />
                      )}
                    </div>
                    {!showDescriptions && (
                      <div className="text-xs text-gray-500 truncate">{info.description}</div>
                    )}
                  </div>
                </motion.button>

                <AnimatePresence>
                  {isExpanded && showDescriptions && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden ml-11 mt-1"
                    >
                      <div className={clsx('p-3 rounded-lg bg-cyber-dark text-sm', info.borderColor, 'border')}>
                        <p className="text-gray-400">{info.description}</p>
                        <div className="mt-2 flex items-center gap-2">
                          <span className="text-xs text-gray-500">Status:</span>
                          <span className={clsx('text-xs', info.color)}>{info.status}</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function LayerBadge({ type, status }: { type: LayerType; status?: LayerInfo['status'] }) {
  const config = layerConfigs[type];
  const displayStatus = status || config.status;

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
        config.bgColor,
        config.color
      )}
    >
      <span>{statusIcons[displayStatus]}</span>
      {config.name}
    </span>
  );
}