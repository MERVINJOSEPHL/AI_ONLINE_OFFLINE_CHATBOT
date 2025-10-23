# 🧠 AI_ONLINE_OFFLINE_CHATBOT

**Date:** October 23, 2025  
**Overview:**  
This project implements a **dual-mode Retrieval-Augmented Generation (RAG)** chatbot, developed as part of the *AI Developer Assignment*. It supports both **online (PDF-based)** and **offline (API-based)** modes, using **FastAPI** as the core framework.

- **PDF Mode (Online):** Uses GPT-4 to answer queries from uploaded PDF documents.  
- **API Mode (Offline):** Uses a locally hosted model (Hugging Face/Ollama) to respond to real-time server metric queries.  

Both modes are exposed via FastAPI routes and verified using Postman collections.

---

## 🚀 Key Features

### 🧩 Task 1: PDF Mode (GPT-RAG)
**Purpose:** Query information from uploaded PDF documents using GPT-4 RAG pipeline.

- **Endpoints:**
  - `POST /upload_pdf` – Upload PDF documents for ingestion.
  - `POST /api/query/pdf` – Query the ingested documents.
- **Example Queries & Responses:**
  - **WebLogic Support:** ✅ Yes (versions: 5.1, 6.x, 7, 8.x, 9.x, 10.x, 11G, 12.x, 14c)
  - **IBM Mainframe:** ❌ No
  - **Oracle DB Monitoring:** ✅ Yes (versions: 7, 8, 9, 10G, 11G, 12c incl. multi-tenant, 19c, 21c)
  - **MSSQL Versions:** 7.0, 2000, 2005, 2008, 2012, 2014, 2016, 2017, 2019, 2022
  - **Open Telemetry:** ❌ No

---

### ⚙️ Task 2: API Mode (Offline Model)
**Purpose:** Query real-time metrics pulled from eG Innovations APIs using an offline model.

- **Endpoints:**
  - `GET /api/realtime_metrics_source` – Fetch server metrics.
  - `POST /api/query/api` – Query metrics using the offline LLM.
- **Example Queries (Server: 10.200.2.192):**
  - **Free Space:** 335,855 MB (last: Oct 23, 2025 10:37:25; CPU 15.2%, status: *normal*)
  - **CPU Utilization:** 22.0% (*normal*)
  - **Memory Usage:** 6,708 MB (*warning*; last: Oct 23, 2025 10:40:36; CPU 22.5%, *normal*)

---

## 📦 Deliverables

- **Codebase:** Modular FastAPI-based Python project (`app/` structure)
- **Dependencies:** `requirements.txt`
- **Samples:** `eg_innovation_complete_task.pdf`
- **Postman Collections:** Pre-configured for both modes (PDF & API)
- **Screenshots:** Postman requests, console logs, and outputs (attached separately)

---

## 🛠️ Setup Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the FastAPI application
uvicorn app.main:app --reload
