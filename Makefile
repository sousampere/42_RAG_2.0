# RAG

# --- Project variables ---

NAME := src
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
	$(PYTHON) -m flake8 src
	$(PYTHON) -m mypy src --strict

debug: install
	$(PYTHON) -m pdb -m src

# --- Automation of running prompts ---

index: install
	clear
	$(PYTHON) -m $(NAME) index --max_chunk_size $(MAX_CHUNK_SIZE)

search: install
	uv run python -m src search "How to setup an OpenAI server ?" --k 5

search_dataset: install
	uv run python -m src search_dataset --dataset_path 'data/datasets/UnansweredQuestions/dataset_code_public.json' --k 10 --save_directory data/output/search_results/UnansweredQuestions/dataset_code_public.json

answer: install
	uv run python -m src answer "What is the type hint for the kv_range_for_decode parameter in the _attention_with_mask method?"

# --- Moulinette ---

moulinette-code: install
	uv run python -m src search_dataset --dataset_path 'data/datasets/UnansweredQuestions/dataset_code_public.json' --k 10 --save_directory data/output/search_results/UnansweredQuestions/dataset_code_public.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results 'data/output/search_results/UnansweredQuestions/dataset_code_public.json' 'data/datasets/AnsweredQuestions/dataset_code_public.json' --k 10

moulinette-docs: install
	uv run python -m src search_dataset --dataset_path 'data/datasets/UnansweredQuestions/dataset_docs_public.json' --k 10 --save_directory data/output/search_results/UnansweredQuestions/dataset_docs_public.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results 'data/output/search_results/UnansweredQuestions/dataset_docs_public.json' 'data/datasets/AnsweredQuestions/dataset_docs_public.json' --k 10
