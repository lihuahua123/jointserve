
time curl http://localhost:9999/generate  -H "Content-Type: application/json"  -d '{
"model": "lora1",
"prompt": "San Francisco is a best free dating site meeting nice single men in san",
"max_tokens": 7,
"temperature": 0
}'
curl http://localhost:9999/generate \
-H "Content-Type: application/json" \
-d '{
"model": "lora2",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'