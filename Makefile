.PHONY: run test lint frontend-build

APP ?= app.main:app
HOST ?= 0.0.0.0
PORT ?= 8666
FRONTEND_DIR ?= ai-agent-chat

run: frontend-build
	@echo "Local: http://127.0.0.1:$(PORT)"
	@echo "UI:    http://127.0.0.1:$(PORT)/agent-ui"
	@echo "Docs:  http://127.0.0.1:$(PORT)/docs"
	@LAN_IP=$$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || ip -4 addr show scope global | awk '/inet / {sub(/\/.*/, "", $$2); print $$2; exit}'); \
	if [ -n "$$LAN_IP" ]; then \
		echo "LAN:   http://$$LAN_IP:$(PORT)"; \
		echo "LAN UI:http://$$LAN_IP:$(PORT)/agent-ui"; \
		echo "LAN API Docs: http://$$LAN_IP:$(PORT)/docs"; \
	fi
	uvicorn $(APP) --host $(HOST) --port $(PORT) --reload --access-log

frontend-build:
	@test -d "$(FRONTEND_DIR)" || { echo "Missing frontend directory: $(FRONTEND_DIR)"; exit 1; }
	@test -f "$(FRONTEND_DIR)/package.json" || { echo "Missing $(FRONTEND_DIR)/package.json"; exit 1; }
	@echo "Building agent UI in $(FRONTEND_DIR)"
	@cd "$(FRONTEND_DIR)" && npm install && npm run build

test:
	pytest

lint:
	python -m compileall app tests
