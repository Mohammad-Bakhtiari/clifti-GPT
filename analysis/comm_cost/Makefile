# Communication-cost supplementary PDF and LaTeX fragments.
#
# All generated artifacts (CSV, PNG, .tex, PDF) land in output/comm_cost/.
# Edit communication_cost.tex in this directory; render_comm_cost_tex.py
# syncs it into the output directory before pdflatex.

REPO_ROOT := ../..
OUTPUT := $(REPO_ROOT)/output/comm_cost
PYTHON := python

.PHONY: all benchmark figures tex pdf clean

all: pdf

benchmark:
	cd $(REPO_ROOT) && $(PYTHON) analysis/comm_cost/comm_cost.py

figures:
	cd $(REPO_ROOT) && $(PYTHON) analysis/comm_cost/plot_comm_cost.py

tex:
	cd $(REPO_ROOT) && $(PYTHON) analysis/comm_cost/render_comm_cost_tex.py

pdf: tex
	cd $(OUTPUT) && pdflatex -interaction=nonstopmode communication_cost.tex
	cd $(OUTPUT) && pdflatex -interaction=nonstopmode communication_cost.tex

clean:
	rm -f $(OUTPUT)/communication_cost.{aux,log,out}
