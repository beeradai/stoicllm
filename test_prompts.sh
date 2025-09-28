#!/usr/bin/env bash
# StoicLLM regression test runner

API_URL=${1:-http://localhost:8000/generate}

echo "�� Running StoicLLM tests against $API_URL"
echo

# Prompts and expected "golden" responses (simplified)
prompts=(
  "Hello StoicLLM, how are you today?"
  "Give me a stoic perspective on dealing with daily stress."
  "Write a short stoic meditation about gratitude."
  "What would Epictetus say about losing your job?"
  "How can I stay calm when criticized by others?"
  "Repeat the word TEST three times."
)

# Expected responses (keep them short/fuzzy for flexibility)
expecteds=(
  "I am a model"
  "stress is perception"
  "gratitude"
  "Epictetus"
  "stay calm"
  "TEST TEST TEST"
)

# Loop through prompts and compare with expected output
for i in "${!prompts[@]}"; do
  prompt="${prompts[$i]}"
  expected="${expecteds[$i]}"

  echo "➡️ Prompt: $prompt"
  response=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$prompt\"}")

  echo "Response: $response"

  if echo "$response" | grep -qi "$expected"; then
    echo "✅ PASS (found \"$expected\")"
  else
    echo "❌ FAIL (expected \"$expected\")"
  fi

  echo -e "---\n"
done
