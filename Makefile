.PHONY: doctor test smoke up down restart status deploy

doctor:
	python -m ops.runtime.doctor

test:
	pytest -q

smoke:
	python -m ops.runtime.smoke

up:
	bash ops/runtime/up.sh

down:
	bash ops/runtime/down.sh

restart:
	bash ops/runtime/restart.sh

status:
	bash ops/runtime/status.sh

deploy:
	bash ops/runtime/deploy.sh
