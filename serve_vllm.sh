#!/bin/bash
# =============================================================================
# Запуск двух vLLM серверов на одной A100 80GB:
# 1. gpt-oss-20b (для эволюции/мутаций) - порт 44002, ~50% GPU памяти
# 2. Llama 3.1 8B (для пациента в MedIQ) - порт 44003, ~35% GPU памяти
# =============================================================================

set -e

# Общие переменные окружения
export TORCHINDUCTOR_CACHE_DIR=/media/ssd-3t/isviridov/vllm_compile_cache
export HF_HOME=/media/ssd-3t/isviridov/models_cache
export HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=1


echo "=============================================="
echo "Starting vLLM servers on A100 80GB"
echo "=============================================="


# -----------------------------------------------------------------------------
# Сервер 1: gpt-oss-20b для эволюции (мутации)
# Порт: 44002
# GPU Memory: ~40GB (0.5)
# max-model-len: 16384 — для prompt мутации (~8.5k токенов) + ответ (~2.5k токенов)
# max-num-seqs: 8 — до 8 параллельных запросов (для 4 мутаций + insights/lineage)
# -----------------------------------------------------------------------------
echo "[1/2] Starting gpt-oss-20b server on port 44002..."
vllm serve openai/gpt-oss-20b \
    --port 44002 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 16384 \
    --max-num-seqs 2 \
    &
EVOLUTION_PID=$!
echo "Evolution model server PID: $EVOLUTION_PID"


# # Подождать немного перед запуском второго сервера
sleep 45


# -----------------------------------------------------------------------------
# Сервер 2: Llama 3.1 8B для MedIQ (пациент/эксперт)
# Порт: 44003
# GPU Memory: ~28GB (0.35)
# max-model-len: 2048 — контекст диалога + facts
# max-num-seqs: 16 — больше параллельных запросов для 4 DAG × ~4 вызова на кейс
# -----------------------------------------------------------------------------
echo "[2/2] Starting Llama 3.1 8B server on port 44003..."
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 44003 \
    --gpu-memory-utilization 0.2 \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    &
LLAMA_PID=$!
echo "Llama server PID: $LLAMA_PID"


# echo ""
# echo "=============================================="
# echo "Both servers starting..."
# echo "  gpt-oss-20b:    http://localhost:44002/v1  (PID: $EVOLUTION_PID)"
# echo "  Llama 3.1 8B:   http://localhost:44003/v1  (PID: $LLAMA_PID)"
# echo "=============================================="
# echo ""
# echo "Waiting for servers to be ready..."
# echo "Press Ctrl+C to stop both servers."

# Ждём завершения (Ctrl+C остановит оба)
wait
