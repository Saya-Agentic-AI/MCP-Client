# Code is present in https://github.com/sayantankpr/MCP-Client.git
# To run this locally - GPU laptop is needed
# Note: This is NON-Docker
# LangGraph + Ollama + MCP client

This is a LangGraph-based replacement for the original Ollama/MCP client. It keeps the enterprise MCP server and PostgreSQL service separate and unchanged.

## Architecture

```text
User
  -> LangGraph ReAct agent (state + tool-call loop + stop conditions)
      -> LangChain MCP adapter
          -> existing MCP server at http://127.0.0.1:8002/mcp
              -> existing customer service/repository
                  -> PostgreSQL
```

The graph uses `MemorySaver` for in-process checkpointing. This is suitable for local learning, but its state is lost when the process exits. It is not a durable production checkpointer.

## Prerequisites

- Python 3.10+ (use a version supported by the installed LangGraph/LangChain packages)
- Ollama installed and running
- Your existing customer MCP server running at `http://127.0.0.1:8002/mcp`
- The selected Ollama model downloaded locally

Example:

```bash
ollama pull gpt-oss:20b
ollama list
```

Choose a model that fits your laptop and supports reliable tool calling. `gpt-oss:20b` can be demanding on memory; set `OLLAMA_MODEL` to a smaller installed model if needed.

## Install

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MCP_SERVER_URL="http://127.0.0.1:8002/mcp"
$env:OLLAMA_HOST="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="gpt-oss:20b"
python client.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MCP_SERVER_URL="http://127.0.0.1:8002/mcp"
export OLLAMA_MODEL="gpt-oss:20b"
python client.py
```

If your MCP server uses another URL, set `MCP_SERVER_URL` accordingly. The server's `/mcp` endpoint and port must match the actual server configuration.

## Try these prompts

- What is the balance of C001?
- Tell me about customer C002.
- Summarize the latest transactions for C005.
- Compare the customer's profile and transaction history, then summarize what is known and what is missing.

Use `/new` to start a fresh in-memory conversation. Use `exit` to stop.

## What changed from the original client?

- `MultiServerMCPClient` discovers MCP tools and adapts them into LangChain tools.
- `ChatOllama` supplies the local LLM.
- `create_react_agent` builds a LangGraph-managed tool-use loop.
- `MemorySaver` maintains conversation state by thread ID while the process runs.
- `recursion_limit` bounds graph execution.

## Important limitations

- This is a **single agent with tool orchestration**, not yet a multi-agent team.
- `MemorySaver` is in-memory only. Use a persistent checkpointer for restart/resume requirements.
- Tool access is not authorization. Add authentication, authorization, audit logging, validation, and approval gates before exposing sensitive or mutating tools.
- The MCP server and database code are intentionally not changed by this client conversion.
- Dependency APIs evolve. If package resolution chooses incompatible versions, use a tested compatible set and pin it in a lock file.
