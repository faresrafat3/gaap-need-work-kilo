'use client';

import { forwardRef, InputHTMLAttributes, TextareaHTMLAttributes, ReactNode } from 'react';
import { clsx } from 'clsx';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  leftElement?: ReactNode;
  rightElement?: ReactNode;
  isFullWidth?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      label,
      error,
      hint,
      leftElement,
      rightElement,
      isFullWidth = true,
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className={clsx('space-y-1.5', isFullWidth && 'w-full')}>
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-gray-300">
            {label}
          </label>
        )}
        <div className="relative">
          {leftElement && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
              {leftElement}
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={clsx(
              'bg-cyber-dark border rounded-lg px-3 py-2 text-white placeholder-gray-500',
              'focus:outline-none focus:ring-2 focus:ring-layer1/50 focus:border-layer1',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              error ? 'border-error' : 'border-layer1/30',
              leftElement && 'pl-10',
              rightElement && 'pr-10',
              isFullWidth && 'w-full',
              className
            )}
            {...props}
          />
          {rightElement && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">
              {rightElement}
            </div>
          )}
        </div>
        {error && <p className="text-xs text-error">{error}</p>}
        {hint && !error && <p className="text-xs text-gray-500">{hint}</p>}
      </div>
    );
  }
);

Input.displayName = 'Input';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  hint?: string;
  isFullWidth?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, hint, isFullWidth = true, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className={clsx('space-y-1.5', isFullWidth && 'w-full')}>
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-gray-300">
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={inputId}
          className={clsx(
            'bg-cyber-dark border rounded-lg px-3 py-2 text-white placeholder-gray-500',
            'focus:outline-none focus:ring-2 focus:ring-layer1/50 focus:border-layer1',
            'disabled:opacity-50 disabled:cursor-not-allowed resize-none',
            error ? 'border-error' : 'border-layer1/30',
            isFullWidth && 'w-full',
            className
          )}
          {...props}
        />
        {error && <p className="text-xs text-error">{error}</p>}
        {hint && !error && <p className="text-xs text-gray-500">{hint}</p>}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <label htmlFor={inputId} className="inline-flex items-center gap-2 cursor-pointer">
        <input
          ref={ref}
          type="checkbox"
          id={inputId}
          className={clsx(
            'w-4 h-4 rounded border-layer1/30 bg-cyber-dark text-layer1',
            'focus:ring-2 focus:ring-layer1/50 focus:ring-offset-2 focus:ring-offset-cyber-dark',
            className
          )}
          {...props}
        />
        {label && <span className="text-sm text-gray-300">{label}</span>}
      </label>
    );
  }
);

Checkbox.displayName = 'Checkbox';

export interface SwitchProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  description?: string;
}

export const Switch = forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, label, description, checked, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <label htmlFor={inputId} className="flex items-center justify-between cursor-pointer">
        {(label || description) && (
          <div>
            {label && <span className="text-sm font-medium text-gray-300">{label}</span>}
            {description && <p className="text-xs text-gray-500">{description}</p>}
          </div>
        )}
        <div className="relative">
          <input ref={ref} type="checkbox" id={inputId} className="sr-only peer" checked={checked} {...props} />
          <div
            className={clsx(
              'w-11 h-6 rounded-full transition-colors',
              checked ? 'bg-layer1' : 'bg-cyber-dark border border-layer1/30',
              className
            )}
          />
          <div
            className={clsx(
              'absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform',
              checked && 'translate-x-5'
            )}
          />
        </div>
      </label>
    );
  }
);

Switch.displayName = 'Switch';
