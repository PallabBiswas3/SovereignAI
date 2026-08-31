FROM python:3.11-alpine
RUN addgroup -g 65534 sandbox 2>/dev/null || true && adduser -D -u 65534 -G sandbox sandbox 2>/dev/null || true
USER 65534:65534
WORKDIR /work
CMD ["python", "--version"]
