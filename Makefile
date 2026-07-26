.PHONY: lint typecheck test run-dashboard build-cpp

lint:
	ruff check .

typecheck:
	mypy src/

test:
	pytest tests/ -v

run-dashboard:
	pwsh scripts/run_dashboard.ps1

build-cpp:
	pwsh scripts/build_cpp.ps1
