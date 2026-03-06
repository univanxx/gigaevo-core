#!/bin/bash
# =============================================================================
# Запуск GigaEvo для задачи gigaevo_mediq_task
# Подключение к обеим моделям (evolution + MedIQ) по API. Redis — локально.
# Отредактируй переменные ниже под свои эндпоинты. Ключ API (если нужен): export OPENAI_API_KEY=...
# Режим алгоритма: best / exploration — см. конец скрипта.
# =============================================================================

set -e

# Redis (локальный)
REDIS_PORT=44008

# Evolution LLM (API)
EVOLUTION_LLM_URL="https://inference.airi.net:46783/v1"
EVOLUTION_MODEL_NAME="Openai/Gpt-oss-120b"
OPENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY3ZDg5YWY5LTUzN2UtNGFhMC1hMGRkLTZiMzI4NjUwZGViNCIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzI0NTc4MDgsImV4cCI6MTc3MzA2MjYwOH0.Pw95tNJ8dgDFifY6htUbnbIdaB-2V93IS3QhAoycJVs"

# MedIQ LLM (API, пациент/врач)
MEDIQ_VLLM_URL="https://inference.airi.net:46783/v1"  # "https://inference.airi.net:46783/v1" -- AIRI, "http://localhost:44005/v1" -- local
MEDIQ_MODEL_NAME="Meta-llama/Llama-3.1-70B-Instruct"

export OPENAI_API_KEY
export MEDIQ_VLLM_URL
export MEDIQ_MODEL_NAME

# Option 1: Очистить Redis перед запуском
echo "Очистка Redis на порту ${REDIS_PORT}..."
redis-cli -p "${REDIS_PORT}" FLUSHDB || echo "Redis не запущен или уже пуст"

# Option 2: Не очищать Redis перед запуском

echo ""
echo "=============================================="
echo "Запуск GigaEvo для gigaevo_mediq_task"
echo "  Redis:     port ${REDIS_PORT}"
echo "  Evolution: ${EVOLUTION_MODEL_NAME} @ ${EVOLUTION_LLM_URL}"
echo "  MedIQ:     ${MEDIQ_MODEL_NAME} @ ${MEDIQ_VLLM_URL}"
echo "=============================================="
echo ""

# =============================================================================
# Параметры (в коде):
# -----------------------------------------------------------------------------
# max_tokens:      Максимум токенов для ответа мутации.
#                  Должно быть < (max-model-len - длина_промпта).
#                  Промпт мутации ~8,500 токенов, ответ ~2,400 токенов.
#                  При max-model-len=16384 для gpt-oss-20b ставим 4096.
#
# stage_timeout:   Таймаут одного этапа DAG (entrypoint = run_mediq на MAX_CASES кейсах).
#                  implicit/scale ~1–2 мин на 100 кейсов; binary ~25–40 мин на 100.
#                  При MAX_CASES=50: binary ~15–20 мин → stage_timeout=1200–1500.
#                  При MAX_CASES=100 и binary: ставить 2400–3600.
#
# dag_timeout:     Таймаут всего DAG (entrypoint + validation + insights).
#                  Обычно 1.2–1.5x от stage_timeout.
#
# max_mutations_per_generation:
#                  Сколько мутаций генерировать параллельно за поколение.
#                  Ограничено GPU памятью для gpt-oss-20b.
#                  При KV cache usage 1-3% можно увеличить до 4-6.
#
# dag_concurrency: Сколько DAG (оценок программ) запускать параллельно.
#                  Каждый DAG = один поток запросов к MedIQ Llama. Должно быть ≤ vLLM --max-num-seqs.
# =============================================================================

# Best (exploitation): elite_selector = FitnessProportionalEliteSelector
python run.py \
    problem.name=gigaevo_mediq_task \
    redis.port="${REDIS_PORT}" \
    llm_base_url="${EVOLUTION_LLM_URL}" \
    model_name="${EVOLUTION_MODEL_NAME}" \
    max_tokens=8192 \
    stage_timeout=2400 \
    dag_timeout=3000 \
    max_mutations_per_generation=4 \
    dag_concurrency=4 \
    redis.resume=true

# Exploration (больше разнообразия стратегий): elite_selector = RandomEliteSelector
# Раскомментировать и закомментировать блок выше при необходимости:
# python run.py \
#     problem.name=gigaevo_mediq_task \
#     algorithm=single_island_exploration \
#     redis.port="${REDIS_PORT}" \
#     llm_base_url="${EVOLUTION_LLM_URL}" \
#     model_name="${EVOLUTION_MODEL_NAME}" \
#     max_tokens=8192 \
#     stage_timeout=1500 \
#     dag_timeout=2000 \
#     max_mutations_per_generation=4 \
#     dag_concurrency=8
