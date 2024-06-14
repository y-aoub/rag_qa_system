# Overview

# Project Overview

This project is a technical assessment that consists of implementing a RAG-based chatbot system to QA data from BioRxiv and eLife.

## Technologies Used

### LangChain

LangChain is used to build the core pipeline for the RAG process, the Chatbot, and their integration. It also handles PDF parsing, document transformation, MemoryBuffer integration, and prompt handling, making it a very handy tool for building such applications.

### Ollama

Ollama was selected due to my computer's limited computing capacity. It simplifies the use of quantized models, improving efficiency without significantly compromising model quality or performance.

- **LLM Choice**: [microsoft/Phi-3-mini-128k-instruct](https://ollama.com/library/phi3:3.8b-instruct) is chosen for its compact size and high performance. It supports a large context window of around 128k tokens, ideal for maintaining a large memory of contextual information in a chat scenario.


### HuggingFace

HuggingFace serves multiple roles in this project:

- **Data and Embedding Model Loading**: Data retrieval and integration of the embedding model are managed through HuggingFace.
  
- **Summarizer Model**: [pszemraj/long-t5-tglobal-base-sci-simplify-elife](https://huggingface.co/pszemraj/long-t5-tglobal-base-sci-simplify), fine-tuned on scientific article summarization tasks. This model is wrapped with [textsum](https://github.com/pszemraj/textsum) to condense XMLs and PDFs into a format suitable for the embedding model's context window.
  
- **Embedding Model**: 'Alibaba-NLP/gte-large-en-v1.5' is selected based on the MTEB Leaderboard criteria for memory usage (1.62 GB), large context window (Max Tokens: 8192), model size (434 million parameters), and retrieval performance.

### Streamlit

Streamlit provides an easy-to-build and user-friendly interface, facilitating easy interaction with the chat application, like this one.


# Data

The dataset consists of scientific articles collected from two primary sources: [BioRxiv](https://www.biorxiv.org/) and [eLife](https://elifesciences.org/). The data is sourced from three distinct repositories:

1. **HuggingFace Dataset (eLife)**:
   - Dataset ID: [pszemraj/scientific_lay_summarisation-elife-norm](https://huggingface.co/datasets/pszemraj/scientific_lay_summarisation-elife-norm/viewer)
   - Data Format: Parquet

2. **BioRxiv API**:
   - Access scientific articles directly from the BioRxiv API.
   - [Documentation](https://api.biorxiv.org/) for accessing the data through the API.
   - Data Format: PDF

3. **eLife GitHub Repository**:
   - XML files extracted from the [eLife GitHub repository](https://github.com/elifesciences/elife-article-xml).
   - Data Format: XML


# Setup

## Creating a Virtual Environment
Create a virtual environment to isolate package dependencies. Use one of the following methods:

- Using Python's built-in venv module:
  ```
  python -m venv [environment_path]
  ```
  
- Using the virtualenv package:
  ```
  virtualenv [environment_path]
  ```

## Activating the Environment
Activate the virtual environment by running:
```
source [environment_path]/bin/activate
```

## Installing Dependencies
Install the required Python packages using:
```
pip install -r requirements.txt
```

# Run

This chatbot can be accessed using two different interfaces: a command-line interface and a web-based interface using Streamlit.

## Running the CLI

You can interact with the chatbot directly from the command line by running:

```bash
python main.py --embedding_device cuda --n_files 10 --build_vector_store
```

- `--embedding_device`: Specifies the device for embeddings (cpu or cuda). Defaults to cpu.

- --`n_files`: Number of PDFs and XMLs to process. Defaults to 5.

- `--build_vector_store`: Boolean flag to build CHroma vector store after processing. Defaults to False.

## Running the Streamlit App

Alternatively, you can use a web-based interface to interact with the chatbot. Run the following command to start the Streamlit app:

```bash
streamlit run app.py -- --embedding_device cuda --n_files 10 --build_vector_store
```

This command runs the Streamlit application with CUDA-enabled GPU for the embedding model, processing 5 PDFs and 5 XMLs, and building the Chroma vector store after fetching, processing, and parsing the raw data.

