#!/bin/bash
# Environment setup script for delta-map belief update experiments.
# Source this file before running any experiments: source setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/.venv/bin/activate"

# Load .env file
set -a
source "${SCRIPT_DIR}/.env"
set +a

# Map LEMMA MaaS proxy credentials to the env vars expected by the ToS codebase.
# The gemini-3-pro config in base_model_config.yaml has base_url pointing to the
# MaaS proxy. The OpenAI SDK client uses OPENAI_API_KEY when no organization is set.
export OPENAI_API_KEY="${LEMMA_MAAS_API_KEY}"
