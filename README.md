## Local LLM RAG with Web page scraped content as context

This is a Dash based web-application which can accept any web URL and enable chat with it based on RAG. The LLM and RAG is without any API calls. That is, it's completely open-source models run locally using Ollama.

![Main Page](/imgs/main_page.png)

### Example 1: Summarize content in URL
![EX1](/imgs/summarize_url.png)

### Example 2: Ask specific question from content in URL
![EX2](/imgs/ask_specific_question.png)

### How to run?
1. You need to install [ollama](https://ollama.com/) for running open-source models locally.

2. Using CLI pull following models:

    `ollama pull llama3`

    `ollama pull mxbai-embed-large`

    `ollama pull mistral`

    `ollama pull nomic-embed-text`

3. Also isntall following python dependencies with pip:

    `pip install langchain`

    `pip install langchain_ollama`

    `pip install dash`

4. Make sure ollama app is running.

5. Run `app.py` file to launch web app.

