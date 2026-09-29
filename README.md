# Excel Link Downloader

This application scans every `.xlsx` file in `data`, extracts Excel hyperlinks and HTTP(S) URLs from cells, and downloads each unique link into `downloaded`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

From the project directory:

```powershell
python download_links.py
```

To inspect the links without downloading:

```powershell
python download_links.py --dry-run
```

Optional directories can be supplied with `--data-dir` and `--output-dir`.

## Build the RAG index

Start PostgreSQL with the `pgvector` extension. The included Compose file creates a local database for development:

```powershell
docker compose up -d postgres
```

Set the connection string:

```powershell
$env:PG_CONN_STR = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
```

Install the RAG dependencies and index the downloaded PDFs into PostgreSQL:

```powershell
python -m pip install -r requirements.txt
python rag_indexer.py index
```

The indexer splits PDF text into 200-word chunks with 30 words of overlap, embeds the PDF text only with `all-MiniLM-L6-v2`, and stores vectors plus structured Excel row metadata in the `rag_documents` PostgreSQL table. Metadata is returned separately from retrieved content and is filterable through both original `excel_*` keys and normalized keys such as `failure_id`, `failure_mode`, `causes`, `technical_effects`, `severity`, `rpn`, `detective_controls`, and `preventive_controls`. It creates the `vector` extension and an HNSW vector index automatically. It matches PDFs to Excel links by the downloaded URL filename and carries all populated Excel fields into each chunk.

Query the index:

```powershell
python rag_indexer.py query "What can cause generator phase imbalance?"
```

Combine vector search with metadata filters:

```powershell
python rag_indexer.py query "What can cause overheating?" --filter "excel_fm_class=Stator Assembly" --filter "excel_severity_1_10=9"
```

The query command returns the top matching chunks, similarity scores, and metadata as JSON for use in an AI prompt builder. The stored metadata includes the source workbook, sheet, row, URL, and all populated Excel fields. Excel metadata keys are normalized to lowercase underscore names for reliable filtering.

## Run five example queries

Run five representative semantic and metadata-filtered queries in one process:

```powershell
python run_example_queries.py
```

The script uses `PG_CONN_STR`, reuses the embedding model for all five queries, and prints the matching chunks with scores and metadata. Use `--top-k 1` to return one result per query.