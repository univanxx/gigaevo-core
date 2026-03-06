#!/bin/bash
# =============================================================================
# Запуск Redis для GigaEvo на порту 44001 с персистентностью.
# Используется для задач (в т.ч. gigaevo_mediq_task).
# =============================================================================

set -e

# Порт и директория для данных Redis
# Кастомный порт: ./serve_redis.sh 44002  (по умолчанию 44001)
REDIS_PORT="${1:-44001}"
REDIS_DIR="/media/ssd-3t/isviridov/alphaevolve/redis-data"

echo "Запуск Redis на порту ${REDIS_PORT}..."
echo "Директория данных: ${REDIS_DIR}"

mkdir -p "${REDIS_DIR}"

# Конфигурация:
# - --dir: куда сохранять dump.rdb / appendonly.aof
# - --appendonly yes: включить AOF для более надёжной персистентности
redis-server \
  --port "${REDIS_PORT}" \
  --dir "${REDIS_DIR}"

