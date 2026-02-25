/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Layer colors
        layer0: '#1a1a2e',
        layer1: '#4a0e78',
        layer2: '#0d47a1',
        layer3: '#2e7d32',
        healing: '#e65100',
        error: '#b71c1c',
        warning: '#f57c00',
        success: '#00c853',
        // Cyber-Noir theme
        cyber: {
          dark: '#0a0a0f',
          darker: '#050508',
          primary: '#00ff88',
          secondary: '#00ccff',
          accent: '#ff00ff',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgb(0 255 136 / 0.5)' },
          '100%': { boxShadow: '0 0 20px rgb(0 255 136 / 0.8)' },
        }
      }
    },
  },
  plugins: [],
}
