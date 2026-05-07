# LangChain & LangGraph Learning Repository

A comprehensive exploration and implementation guide for LangChain and LangGraph frameworks, covering fundamental concepts to advanced agent architectures and RAG systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Module Breakdown](#module-breakdown)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Requirements](#requirements)

## 🎯 Overview

This repository contains educational projects and practical implementations demonstrating:

- **LangChain** fundamentals and advanced patterns
- **LangGraph** for building stateful agent workflows
- **RAG (Retrieval Augmented Generation)** systems
- **Agent design patterns** and tool integration
- **Output parsing** and structured data handling
- **Real-time chat** and streaming capabilities
- **MCP (Model Context Protocol)** server implementations

## 📁 Project Structure

### Core Learning Modules

#### `01_LLMInteraction/`
- **Purpose**: Introduction to basic LLM interactions
- **Files**: 
  - `main.py` - Basic LLM interaction patterns
  - `main.ipynb` - Jupyter notebook walkthrough
  - `app.py` - Application example

#### `02_Chatbot/`
- **Purpose**: Building chatbot applications with message placeholders and prompt templates
- **Files**:
  - `chatbotmessageplaceholder.py` - Message placeholder patterns
  - `chatbotprompttemplate.py` - Prompt template design
  - `main.py` - Complete chatbot implementation

#### `03_structuredOutput/`
- **Purpose**: Handling and generating structured outputs
- **Files**:
  - `main.py` - Core structured output patterns
  - `pydantic_structured_output.py` - Pydantic model integration
  - `advanced_structured_output.py` - Advanced techniques

#### `04_OutputParsers/`
- **Purpose**: Various output parsing strategies
- **Files**:
  - `strOutputParsers.py` - String output parsing
  - `pydanticparser.py` - Pydantic parser implementation
  - `structuredOutputParser.py` - Structured output parsing

#### `05_chains/`
- **Purpose**: Chain patterns and compositions
- **Files**:
  - `llm.chains.py` - Basic LLM chains
  - `sequential.chains.py` - Sequential chain execution
  - `parallel.chains.py` - Parallel chain execution
  - `conditional.chains.py` - Conditional chain logic

#### `06_RAG/`
- **Purpose**: Retrieval Augmented Generation implementation
- **Files**:
  - `main.py` - Core RAG implementation
  - `app.py` - RAG application
  - `pdf_loader.ipynb` - PDF document loading
  - `document.ipynb` - Document processing examples
  - **data/**: Sample documents (PDFs, text files)
  - **vectorstore/**: Chroma vector database storage

#### `07_tools/`
- **Purpose**: Tool definition and integration
- **Files**:
  - `1.tools.py` - Basic tool definitions
  - `2.structured_tool.py` - Structured tool implementation
  - `3.toolkits.py` - Tool collection patterns
  - `tool_binding.py` - Tool binding mechanisms

#### `08_langgraph/`
- **Purpose**: LangGraph framework implementations
- **Files**:
  - `1.main.py` - Basic graph implementation
  - `2.simple_graph.ipynb` - Simple graph patterns
  - `3._graph_with_tools.ipynb` - Tool-integrated graphs
  - `4_chatbot.ipynb` - Chatbot using LangGraph
  - `5_tool_call.ipynb` - Tool calling patterns
  - `6_chatbot_memory.py` - Memory-enhanced chatbots

### Advanced Implementations

#### `Agent_and_tools/`
- **Purpose**: Advanced agent design patterns
- **Files**:
  - `agent_and_tools_basics.py` - Agent fundamentals
  - `agents_react_chat.py` - ReAct pattern implementation
  - `dynamic_model.py` - Dynamic agent models
  - `static_model.py` - Static agent models
  - `human-in-the-loop.py` - Human-in-the-loop agents
  - `create_agents.ipynb` - Agent creation guide

#### `GenAI_Agents/`
- **Purpose**: Advanced AI agents for specific domains
- **Files**:
  - `memory_enhanced_conversational_agent.ipynb` - Conversational agents with memory
  - `simple_question_answering_agent.ipynb` - QA agents
  - `simple_data_analysis_agent_notebook.ipynb` - Data analysis agents
  - `scientific_paper_agent_langgraph.ipynb` - Academic paper processing
  - `Academic_Task_Learning_Agent_LangGraph.ipynb` - Educational agents
  - `ShopGenie.ipynb` - E-commerce agent example
  - `langgraph-tutorial.ipynb` - LangGraph tutorial
  - `mcp-tutorial.ipynb` - MCP integration tutorial
  - **data/**: Domain-specific data files

#### `RAG/`
- **Purpose**: RAG system implementations and variations
- **Files**:
  - `1a_rag_basics.py` - Basic RAG implementation
  - `1b_rag_basics.py` - RAG variations
  - `2a_rag_basics_metadata.py` - RAG with metadata handling
  - `2b_rag_basics_metadata.py` - Advanced metadata patterns
  - `rag_text_splitting_deep_dive.py` - Text chunking strategies
  - `rag_web_scrape_basic.py` - Web-based RAG
  - **books/**: Sample documents for RAG

#### `RAGSystems/`
- **Purpose**: Domain-specific RAG implementations
- **Files**:
  - `youtube_rag.py` - YouTube content RAG system

### Basic Examples

#### `Basic/`
- **Purpose**: Fundamental chat and conversation examples
- **Files**:
  - `main.py` - Basic setup
  - `chat.model.basic.py` - Basic chat model
  - `chat.model.basic_conversion.py` - Basic conversion patterns
  - `chat.mode.realTime.con.py` - Real-time conversation

#### `chat_model/`
- **Purpose**: Chat model implementations
- **Files**:
  - `chat.model.basic.py` - Basic chat model
  - `chat.model.basic_conversation.py` - Conversation patterns
  - `chat.model.realTime.conversation.py` - Real-time conversations

#### `Streaming/`
- **Purpose**: Streaming response handling
- **Files**:
  - `stream1.py` - Basic streaming implementation

#### `Text Summarizer/`
- **Purpose**: Text summarization applications
- **Files**:
  - `main.py` - Text summarizer implementation

### Utility Modules

#### `batch/`
- **Purpose**: Batch processing examples
- **Files**:
  - `batch_01.py` - Batch operation patterns

#### `simple-add-math-operation/` & `simple-app/`
- **Purpose**: Basic utility applications

### MCP Servers

#### `mcp/`
- **Purpose**: Model Context Protocol server implementation
- **Files**:
  - `main.py` - MCP server setup
  - **tool/**: MCP tool definitions

#### `mcp-crypto-server/`
- **Purpose**: Cryptocurrency-specific MCP server
- **Files**:
  - `main.py` - Server entry point
  - `mcp_server.py` - Server implementation
  - `pyproject.toml` - Project configuration

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- API keys for LLM services (OpenAI, etc.)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd langchain-and-langgraph
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For RAG module:
   ```bash
   cd 06_RAG
   pip install -r requirements.txt
   ```

## 📦 Requirements

Key dependencies (see `requirements.txt` for complete list):

- `langchain` - Core LangChain framework
- `langgraph` - State machine framework for agents
- `openai` - OpenAI API integration
- `pydantic` - Data validation and serialization
- `chromadb` - Vector database for RAG
- `jupyter` - Notebook environment
- `python-dotenv` - Environment variable management

## 💡 Key Features

### LangChain
- ✅ LLM interaction patterns
- ✅ Prompt engineering and templates
- ✅ Output parsing and validation
- ✅ Chain composition (sequential, parallel, conditional)
- ✅ Memory management
- ✅ Tool integration

### LangGraph
- ✅ Stateful agent workflows
- ✅ Tool-augmented agents
- ✅ Human-in-the-loop patterns
- ✅ Persistent memory and state
- ✅ Conditional routing and branching

### RAG Systems
- ✅ Document loading and processing
- ✅ Text chunking and splitting strategies
- ✅ Vector embeddings and similarity search
- ✅ Metadata-aware retrieval
- ✅ Web scraping integration

### Agents
- ✅ ReAct pattern implementation
- ✅ Dynamic and static agent models
- ✅ Tool calling and binding
- ✅ Agent memory and context
- ✅ Domain-specific agents

## 🔧 Usage Examples

### Basic LLM Interaction
```bash
python 01_LLMInteraction/main.py
```

### Running a Chatbot
```bash
python 02_Chatbot/main.py
```

### Building a RAG System
```bash
cd 06_RAG
python main.py
```

### Creating a LangGraph Agent
```bash
jupyter notebook 08_langgraph/2.simple_graph.ipynb
```

### Running an Advanced Agent
```bash
jupyter notebook GenAI_Agents/memory_enhanced_conversational_agent.ipynb
```

## 📚 Learning Path

Recommended order for learning:

1. **Foundations** (Week 1)
   - `01_LLMInteraction/` - Basic LLM calls
   - `02_Chatbot/` - Chat interfaces
   - `03_structuredOutput/` - Structured responses

2. **Core Concepts** (Week 2-3)
   - `04_OutputParsers/` - Output handling
   - `05_chains/` - Chain patterns
   - `07_tools/` - Tool definitions

3. **Advanced Topics** (Week 4-5)
   - `06_RAG/` - Retrieval systems
   - `08_langgraph/` - Graph-based agents
   - `Agent_and_tools/` - Agent patterns

4. **Specialized Applications** (Week 6+)
   - `GenAI_Agents/` - Domain-specific agents
   - `RAGSystems/` - Advanced RAG implementations
   - `mcp/` - MCP integration

## 🤝 Contributing

This is an educational repository. Feel free to:
- Extend existing examples
- Add new implementations
- Improve documentation
- Share feedback and improvements

## 📝 Notes

- Check `chatbot_history.txt` for chat interaction logs
- Each module contains independent examples - can be run separately
- Some modules require API keys (configure in `.env`)
- Jupyter notebooks contain detailed walkthroughs
- RAG module includes vector store in `chroma_db/` directory

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Last Updated**: May 2026

For questions or improvements, refer to specific module README files or check the inline documentation in code files.
