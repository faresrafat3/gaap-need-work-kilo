'use client'

import { motion } from 'framer-motion'
import { Eye, Compass, Brain, Zap } from 'lucide-react'
import { useGAAPStore } from '@/lib/store'

const stages = [
  { key: 'observe', label: 'مراقبة', labelEn: 'Observe', icon: Eye, color: '#8b5cf6' },
  { key: 'orient', label: 'توجيه', labelEn: 'Orient', icon: Compass, color: '#6366f1' },
  { key: 'decide', label: 'قرار', labelEn: 'Decide', icon: Brain, color: '#a855f7' },
  { key: 'act', label: 'تنفيذ', labelEn: 'Act', icon: Zap, color: '#7c3aed' },
]

export function OODAVisualization() {
  const { oodaState } = useGAAPStore()

  return (
    <div className="relative w-full h-64 flex items-center justify-center">
      {/* Background ring */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-48 h-48 rounded-full border border-primary/20" />
        <div className="absolute w-56 h-56 rounded-full border border-primary/10" />
        <div className="absolute w-40 h-40 rounded-full border border-primary/30" />
      </div>

      {/* Center glow */}
      <motion.div
        className="absolute w-20 h-20 rounded-full bg-gradient-to-br from-primary/30 to-accent/30 blur-xl"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.5, 0.8, 0.5],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      {/* OODA stages in a circle */}
      {stages.map((stage, index) => {
        const angle = (index * 90 - 90) * (Math.PI / 180)
        const x = Math.cos(angle) * 80
        const y = Math.sin(angle) * 80
        const Icon = stage.icon
        const stageState = oodaState[stage.key as keyof typeof oodaState]
        const isActive = stageState.status === 'processing'
        const isComplete = stageState.status === 'complete'

        return (
          <motion.div
            key={stage.key}
            className="absolute"
            style={{
              left: `calc(50% + ${x}px)`,
              top: `calc(50% + ${y}px)`,
              transform: 'translate(-50%, -50%)',
            }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: index * 0.1, duration: 0.5 }}
          >
            <motion.div
              className={`relative flex flex-col items-center gap-2 p-3 rounded-xl ${
                isActive ? 'bg-primary/20' : isComplete ? 'bg-primary/10' : 'bg-card'
              } border border-primary/20`}
              animate={isActive ? {
                boxShadow: [
                  `0 0 0px ${stage.color}40`,
                  `0 0 20px ${stage.color}60`,
                  `0 0 0px ${stage.color}40`,
                ],
              } : {}}
              transition={{ duration: 1, repeat: isActive ? Infinity : 0 }}
            >
              <div
                className={`p-2 rounded-lg ${
                  isActive ? 'bg-primary/30' : 'bg-secondary'
                }`}
                style={{ color: stage.color }}
              >
                <Icon className="w-5 h-5" />
              </div>
              <div className="text-center">
                <div className="text-xs font-medium text-foreground">
                  {stage.label}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {stage.labelEn}
                </div>
              </div>
              
              {/* Status indicator */}
              <motion.div
                className={`absolute -top-1 -right-1 w-2 h-2 rounded-full ${
                  isActive ? 'bg-yellow-500' : isComplete ? 'bg-green-500' : 'bg-muted-foreground'
                }`}
                animate={isActive ? { scale: [1, 1.3, 1] } : {}}
                transition={{ duration: 0.5, repeat: isActive ? Infinity : 0 }}
              />
            </motion.div>
          </motion.div>
        )
      })}

      {/* Center label */}
      <div className="absolute z-10 flex flex-col items-center">
        <div className="text-lg font-bold gradient-text">OODA</div>
        <div className="text-xs text-muted-foreground">حلقة القرار</div>
      </div>

      {/* Connecting lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {stages.map((stage, index) => {
          const angle = (index * 90 - 90) * (Math.PI / 180)
          const x = Math.cos(angle) * 80
          const y = Math.sin(angle) * 80
          const nextIndex = (index + 1) % 4
          const nextAngle = (nextIndex * 90 - 90) * (Math.PI / 180)
          const nextX = Math.cos(nextAngle) * 80
          const nextY = Math.sin(nextAngle) * 80

          return (
            <motion.line
              key={`line-${stage.key}`}
              x1={`calc(50% + ${x}px)`}
              y1={`calc(50% + ${y}px)`}
              x2={`calc(50% + ${nextX}px)`}
              y2={`calc(50% + ${nextY}px)`}
              stroke="url(#gradient)"
              strokeWidth="1"
              strokeDasharray="4 4"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.3 }}
              transition={{ delay: 0.5 + index * 0.2, duration: 0.5 }}
            />
          )
        })}
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#6366f1" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  )
}
