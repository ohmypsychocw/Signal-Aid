#!/usr/bin/env bash
# Set up Signal Aid on an EC2 GPU instance (target: g4dn.xlarge with an AWS Deep Learning
# Base GPU AMI, Ubuntu). Run from the repo root.
#
# The Decider server has NO authentication: it binds to 127.0.0.1 only.
# Do not open port 8099 in the security group.
#
#   CHECKPOINT=StrandsAgents/strands-decider-2B-hobson-v19 bash scripts/ec2_setup.sh   # older checkpoint
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-StrandsAgents/strands-decider-2B-qwen3.5-v1-2610}"
DEVICE="${DEVICE:-cuda}"
# Pinned to the strands-decider commit this project was built against.
DECIDER_REF="${DECIDER_REF:-930e6d5}"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider@${DECIDER_REF}"
pip install -e ".[dev]"

# Start the Decider server in the background (the first start downloads the weights).
nohup strands-decider serve "${CHECKPOINT}" --device "${DEVICE}" --port 8099 > decider.log 2>&1 &
echo "Waiting for Decider (${CHECKPOINT})..."
until curl -s http://127.0.0.1:8099/health > /dev/null; do sleep 5; done
curl -s http://127.0.0.1:8099/health; echo

echo "Ready. Next:"
echo "  export AWS_REGION=us-east-1   # Bedrock model defaults to us.anthropic.claude-haiku-4-5-20251001-v1:0"
echo "  python examples/quickstart.py"
echo "  python -m eval.evaluate --lang en && python -m eval.evaluate --lang es"
echo "  python -m signal_aid.pipeline --out web/replay.json"
echo "  python scripts/build_web.py && bash scripts/deploy_web.sh <bucket-name>"
