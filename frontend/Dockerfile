FROM node:22-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:22-alpine AS runner
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
WORKDIR /app
RUN addgroup -S sovereign && adduser -S sovereign -G sovereign
COPY --from=builder --chown=sovereign:sovereign /app/.next/standalone ./
COPY --from=builder --chown=sovereign:sovereign /app/.next/static ./.next/static
USER sovereign
EXPOSE 3000
CMD ["node", "server.js"]
