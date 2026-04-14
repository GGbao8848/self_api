.PHONY: run test lint

APP ?= app.main:app
HOST ?= 0.0.0.0
PORT ?= 8666

run:
	@echo "Local: http://127.0.0.1:$(PORT)"
	@LAN_IP=$$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || ip -4 addr show scope global | awk '/inet / {sub(/\/.*/, "", $$2); print $$2; exit}'); \
	if [ -n "$$LAN_IP" ]; then echo "LAN:   http://$$LAN_IP:$(PORT)"; fi
	uvicorn $(APP) --host $(HOST) --port $(PORT) --reload --access-log

test:
	pytest

lint:
	python -m compileall app tests
