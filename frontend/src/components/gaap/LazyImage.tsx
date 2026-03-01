'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { useLazyLoad } from '@/hooks/useLazyLoad'

interface LazyImageProps {
  src: string
  alt: string
  className?: string
  placeholder?: string
}

export function LazyImage({ src, alt, className, placeholder }: LazyImageProps) {
  const [ref, isVisible] = useLazyLoad<HTMLDivElement>({ triggerOnce: true })
  const [isLoaded, setIsLoaded] = useState(false)

  return (
    <div
      ref={ref}
      className={cn(
        'relative overflow-hidden bg-muted',
        className
      )}
    >
      {isVisible && (
        <img
          src={src}
          alt={alt}
          className={cn(
            'w-full h-full object-cover transition-opacity duration-300',
            isLoaded ? 'opacity-100' : 'opacity-0'
          )}
          onLoad={() => setIsLoaded(true)}
        />
      )}
      {!isLoaded && placeholder && (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
          {placeholder}
        </div>
      )}
    </div>
  )
}
