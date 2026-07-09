# RAG

# --- Project variables ---

NAME := ratm
VENV := .venv
PYTHON := $(VENV)/bin/python
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

PROJECT_START_DATE=2026-06-04
PROJECT_NAME=42 RAG
AUTHOR=sousampere
GITHUB=sousampere/42_RAG_2.0

ARGV=
MAX_CHUNK_SIZE=2000

# --- Colors ---

GREEN   := \033[0;32m
RED     := \033[0;31m
YELLOW  := \033[0;33m
BLUE    := \033[0;34m
MAGENTA := \033[0;35m
CYAN    := \033[0;36m
RESET   := \033[0m

ECHO    := echo -e

# --- Rules ---

install:
	@printf "\033[2J\033[H"
	@printf "$(YELLOW)╔════════════════════════════════════════════════════════════════╗\n"
	@printf "$(YELLOW)║                                                                ║\n"
	@printf "$(YELLOW)║  44  44    2222    $(GREEN)Made by $(AUTHOR) $(YELLOW)\n"
	@printf "$(YELLOW)║  44  44   22  22   Project: $(CYAN)$(PROJECT_NAME) $(YELLOW)\n"
	@printf "$(YELLOW)║  444444      22    Started in: $(CYAN)$(PROJECT_START_DATE) $(YELLOW)\n"
	@printf "$(YELLOW)║      44     22     Github: $(CYAN)$(GITHUB) $(YELLOW)\n"
	@printf "$(YELLOW)║      44   222222                                               ║\n"
	@printf "$(YELLOW)║                                                                ║\n"
	@printf "$(YELLOW)╚════════════════════════════════════════════════════════════════╝\n"
	@printf "\033[3;66H║"
	@printf "\033[4;66H║"
	@printf "\033[5;66H║"
	@printf "\033[6;66H║"
	@printf "\033[7;66H║"
	@printf "\033[8;66H║"
	@printf "\033[9;80H\n"
	@printf "$(CYAN)[Installation]$(RESET) ➡️  Synchronizing uv\n"
	uv sync

run: install
	$(PYTHON) -m $(NAME) $(ARGV)

lint: install
	$(PYTHON) -m flake8 src
	$(PYTHON) -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	$(PYTHON) flake8 src ratm.py
	$(PYTHON) mypy src ratm.py --strict

debug: install
	$(PYTHON) -m pdb ratm.py

index: install
	$(PYTHON) -m $(NAME) index --max_chunk_size $(MAX_CHUNK_SIZE)