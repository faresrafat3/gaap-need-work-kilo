'use client';

import { HTMLAttributes, forwardRef } from 'react';
import { clsx } from 'clsx';

export type BadgeVariant = 
  | 'default' 
  | 'success' 
  | 'warning' 
  | 'error' 
  | 'info' 
  | 'layer1' 
  | 'layer2' 
  | 'layer3';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: 'sm' | 'md' | 'lg';
  dot?: boolean;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-gray-500/20 text-gray-300',
  success: 'bg-success/20 text-success',
  warning: 'bg-warning/20 text-warning',
  error: 'bg-error/20 text-error',
  info: 'bg-blue-400/20 text-blue-400',
  layer1: 'bg-layer1/20 text-purple-300',
  layer2: 'bg-layer2/20 text-blue-300',
  layer3: 'bg-layer3/20 text-green-300',
};

const sizeStyles = {
  sm: 'px-1.5 py-0.5 text-xs',
  md: 'px-2 py-0.5 text-xs',
  lg: 'px-2.5 py-1 text-sm',
};

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', size = 'md', dot, children, ...props }, ref) => (
    <span
      ref={ref}
      className={clsx(
        'inline-flex items-center gap-1.5 font-medium rounded',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {dot && (
        <span
          className={clsx(
            'w-1.5 h-1.5 rounded-full',
            variant === 'success' && 'bg-success',
            variant === 'warning' && 'bg-warning',
            variant === 'error' && 'bg-error',
            variant === 'info' && 'bg-blue-400',
            (!variant || variant === 'default') && 'bg-gray-400',
            variant.startsWith('layer') && 'bg-current'
          )}
        />
      )}
      {children}
    </span>
  )
);

Badge.displayName = 'Badge';