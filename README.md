# Learning HuggingFace Agents

This repository contains examples and experiments with different AI agent frameworks, focusing on:

- LlamaIndex
- Smolagents
- LangChain

## Demonstrations

### Code Agent (smolagent_codeagents.py)

A basic example demonstrating:

- Using Smolagents' `CodeAgent` with Ollama backend
- Implementing a simple calculator tool,
- Integrating DuckDuckGo search capability

### Tool Agent (smolagent_toolagents.py)

Another example showing:

- Using Smolagents' `ToolCallingAgent` with Ollama backend
- Similar calculator and search capabilities, adding image generation capability load from HuggingFace Hub

### Multi-Agent System (smolagent_multiagent.py)

Demonstrates agent collaboration:

- Main `CodeAgent` managing a specialized `ToolCallingAgent`
- Web agent focused on information retrieval
- Shows how to compose agents for complex tasks

### RAG with Agents (smolagent_rag.py)

Implements a Retrieval-Augmented Generation system:

- Custom `RetrieverTool` using BM25 retrieval
- Document processing with LangChain
- Semantic search over structured data (CSV)
- Enhanced response handling for missing information

### Example Workflow (llamaindex_workflow.py)

Demonstrates a multi-step workflow using LlamaIndex, handling events and processing results through a state machine.

### Human Interaction (llamaindex_human.py)

An example showcasing how to interact with LlamaIndex using human-like responses and processing.

### Simple Example (llamaindex_simple.py)

A straightforward example demonstrating basic functionalities of LlamaIndex without complex setups.

### Tool Integration (llamaindex_tool.py)

Shows how to integrate various tools with LlamaIndex for enhanced capabilities.

### Advanced Workflow (llamaindex_workflow_2.py)

An advanced example illustrating a more complex workflow with LlamaIndex, including additional features and integrations.

### Spam Email Agent (langchain_firstagent.py)

A simple example of an agent to classify emails as spam or not using LangChain.

Example usage:

```bash
# Using Code Agent
python smolagent_codeagents.py "Calculate 1 + two multiply 3"
python smolagent_codeagents.py "What time is it now?"

# Using Tool Agent
python smolagent_toolagents.py "What is the capital of France?"
python smolagent_toolagents.py "Calculate 1 + two multiply 3"

# Using Multi-Agent System
python smolagent_multiagent.py "Find information about AI agents"

# Using RAG System
python smolagent_rag.py "Give me email of Carol Williams"
```

## Setup

1. Install repo dependencies with uv:

```bash
uv pip install -e .
```

2. Set up Ollama locally (required for the simple_toolagents.py example)

3. Create a `.env` file with necessary API keys (if required)

## Contributing

Feel free to contribute more examples or improvements to existing ones.
