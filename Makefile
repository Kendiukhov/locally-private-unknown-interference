PYTHON ?= python

.PHONY: test quick experiments figures table paper finalize reproduce verify

test:
	$(PYTHON) -m pytest -q

quick: test
	PYTHONPATH=src $(PYTHON) scripts/run_experiments.py --quick --output tmp/quick_artifact
	MPLBACKEND=Agg PYTHONPATH=src $(PYTHON) scripts/make_figures.py \
		--results tmp/quick_artifact/summary --output tmp/quick_artifact/figures
	PYTHONPATH=src $(PYTHON) scripts/make_tables.py \
		--summary tmp/quick_artifact/summary/benchmark.csv \
		--output tmp/quick_artifact/generated_results.tex

experiments:
	PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

figures:
	MPLBACKEND=Agg PYTHONPATH=src $(PYTHON) scripts/make_figures.py

table:
	PYTHONPATH=src $(PYTHON) scripts/make_tables.py

paper: table
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

finalize:
	PYTHONPATH=src $(PYTHON) scripts/finalize_results.py

reproduce: test experiments figures paper

verify: paper finalize
	PYTHONPATH=src $(PYTHON) scripts/verify_artifact.py

