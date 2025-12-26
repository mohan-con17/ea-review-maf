# EA Review MAF (Multi‑Agent Framework)

## Overview
**EA Review MAF** is a working implementation of an **Architecture Review system** built using a **Python-based multi-agent framework**.  
The system orchestrates multiple specialized agents to collaboratively evaluate and review enterprise architecture artifacts, exposed via a **FastAPI** service.

This project is intended for local development and experimentation with agent-driven architecture review workflows.

---

## Key Features
- Multi-agent architecture review framework
- Python-based agent orchestration
- FastAPI backend for service exposure
- Modular and extensible design
- Hot-reload enabled for rapid development

---

## Tech Stack
- **Python 3.12+**
- **FastAPI**
- **Uvicorn**
- **Multi-Agent Framework (Python)**
- **Virtualenv** for dependency isolation

---

## Prerequisites
Ensure the following are installed on your local machine:
- Python 3.9 or higher
- Git
- Virtualenv (comes bundled with modern Python versions)

---

## Steps to Run Locally

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ea-review-maf
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the FastAPI Service
```bash
uvicorn main:app --reload --log-level info
```

---

## Accessing the Application
Once the service is running, access:

- **API Base URL:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## Project Structure (High-Level)
```
ea-review-maf/
│── main.py              # FastAPI application entry point
│── requirements.txt     # Python dependencies
│── agents/              # Multi-agent implementations
│── services/            # Core review logic
│── utils/               # Shared utilities
│── README.md            # Project documentation
```

---

## Development Notes
- The `--reload` flag enables automatic server reload on code changes.
- Logs are set to `info` level for better runtime visibility.
- The architecture is designed to support easy onboarding of new agents.

---

## Future Enhancements
- Agent performance metrics and tracing
- UI dashboard for review visualization
- Configurable agent pipelines
- Cloud deployment support

---

## License
This project is currently intended for internal and experimental use.  
Add an appropriate license if you plan to distribute or open-source the project.

---

## Maintainer
**Senior Software Engineering Team**  
Enterprise Architecture & AI Platforms
