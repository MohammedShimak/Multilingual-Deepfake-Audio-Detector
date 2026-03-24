/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#4f46e5',
        'primary-hover': '#4338ca',
        secondary: '#ec4899',
        success: '#10b981',
        danger: '#ef4444',
        text: '#f8fafc',
        'text-muted': '#94a3b8',
      }
    },
  },
  plugins: [],
}
