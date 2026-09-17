
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.{ts,tsx}",
    "./chat/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./graph/**/*.{ts,tsx}",
    "./ingestion/**/*.{ts,tsx}",
    "./layout/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
      },
      keyframes: {
        loading: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(300%)' },
        }
      },
      animation: {
        loading: 'loading 1.5s infinite linear',
      }
    },
  },
  plugins: [],
}
