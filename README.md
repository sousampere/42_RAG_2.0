*This project has been created as part of the 42 curriculum by gtourdia*

# 42 RAG
A 42 project made by gtourdia, about creating a RAG.

<div align="center">

![42Mulhouse](https://github.com/sousampere/sousampere/blob/main/assets/RAG_banner.png?raw=true)

</div>

# 📝​ Description

This project is about creating a RAG system as a python CLI. It is driven by three main CLI commands :
- index : Index the repository located in ./data/raw
- search : Retrieve the best (k) sources from a user's query
- answer : Answer a user's query using an LLM connected to the search results of the retriever

With all of this, we can also use the commands `search_dataset` and `answer_dataset` to process a whole dataset of questions.

# 💻 Instructions

Python, UV and Make are needed to run this program.

## Installation

```bash
make install
```

## Index

```bash
uv run python -m src index --max_chunk_size 2000
```

## Search for a query

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

## Search a whole dataset of queries

```bash
uv run python -m src search_dataset --dataset_path 'data/datasets/UnansweredQuestions/dataset_docs_public.json' --k 10 --save_directory 'data/output/search_results/UnansweredQuestions'
```

## Answer a query

```bash
uv run python -m src answer "What is the type hint for the kv_range_for_decode parameter in the _attention_with_mask method?" --k 5
```

## Answer a whole dataset of queries

```bash
uv run python -m src answer_dataset --student_search_results_path 'data/output/search_results/UnansweredQuestions/dataset_docs_public.json' --save_directory 'data/output/search_results_and_answer/AnsweredQuestions'
```

# 🔧​​ Ressources

I used 42 user [kebertra / @KeroBeros68](https://github.com/KeroBeros68)'s Obsidian vault as a way to gather informations about the creation of a bm25s + langchain retriever.

The rest are simple web research about tqdm, how to use an LLM using huggingface, etc.

# 🔗​ General Software Architecture

I set up a simple but very efficient setup :

- RagCLI class to handle the CLI (used as Python Fire CLI)
- RagProcessor class to distribute tasks to the Retriever class and the LLMAugmenter class
- Evaluator to manage evaluation

# ⛏️ Chunking strategy

I used LangchainCommunity's built-in Python and Markdown text splitters.

# 🔍 Retrieval method

The BM25 algorithm was used (BM25s module) to index and retrieve data.

# 📈 Performance analysis

The retrieving was not enough at one point. I decided to add some chunking overlap (from 5% to 15%) and improved my results to have exactly 80% of good retrieving with a recall@5.

# 🎨 Design decisions

The single-responsability principe was what I aimed.

# 🧵 Challenges faced

I had some pain with calculating the overlap by my own, but ended up finding a formula with Google Gemini, and then implemented it easily.

# 👍 Example usage

This RAG can be used to retrieve information over any private data that a normal LLM wouldn't know.

# 🔼 Bonus

As it is a bonus in the project, I made a small API that can be interracted.

```bash
make start-api
```


# 🚀 Made by

[gtourdia / @sousampere](https://github.com/sousampere)


![Logo](https://github.com/sousampere/sousampere/blob/main/42mulhouse.png?raw=true)
