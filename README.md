# Introduction

OncoGraphRAG leverages NLP to extract relationships from cancer-focused
textual data and integrates this information into a knowledge graph to
support exploratory analysis. It features an interactive chat system for
querying the graph.
The project is in early development and is primarily available for
experimentation and prototyping.

# Installation

This project consists of four main components: GraphDB for the knowledge graph and a Streamlit app for
interactive chat. You can set up OncoGraphRAG manually as described below.

### Prerequisites

- Python (version 3.9 or newer) and Node.js must be installed on your computer.
- A running instance of GraphDB. Download and install
  from [GraphDB](https://www.ontotext.com/products/graphdb/download/) official site.
- An OpenAI API key for accessing OpenAI's services. Once you have your API key,
  set it as an environment variable named `OPENAI_API_KEY`:
- A Huggingface API token for accessing Huggingface models.

      export OPENAI_API_KEY="sk-...."
      export HUGGINGFACEHUB_API_TOKEN="hf_...."

### GraphDB Configuration

1. Launch GraphDB and create a new repository (for example, `oncograph_repo`).

2. Once the repository is created, you can populate the knowledge graph with
   cancer-related data by uploading the pre-configured RDF file available in
   this repository. You can find the RDF file at `data/OncoGraphRAG_ont.rdf`.
   Use the upload_schema.py and relation_upload.py scripts to upload the schema and relations as required.

3. The knowledge graph is now ready to be queried using the Streamlit app.

### Backend Setup

1. Navigate to the backend directory.
2. Create a virtual environment using Python 3.10.
3. Install the required Python packages from `backend/requirements.txt`.

   `pip install -r requirements.txt`

### Streamlit App Setup

1. Navigate to the backend directory.
2. Start the Streamlit app:

   `streamlit run streamlit_app.py`

3. Note the URL where the Streamlit app is running
   (typically http://localhost:8501).
