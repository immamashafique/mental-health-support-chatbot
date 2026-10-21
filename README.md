# Mental Health Support Chatbot

An AI-powered Mental Health Support Chatbot built using Python, Streamlit, Hugging Face Transformers, LangChain, and FAISS.

## Features

- AI-powered conversational support
- PDF-based question answering using RAG
- Document processing and text splitting
- Semantic search using FAISS
- Conversation memory
- LoRA/PEFT fine-tuning implementation
- Interactive web interface using Streamlit

## Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- LangChain
- FAISS
- Sentence Transformers
- PEFT (LoRA)
- PyPDF

## Project Structure

```text
mental-health-support-chatbot/
├── main.py
├── requirements.txt
├── README.md
├── mental_health_data.jsonl
└── mental_health.pdf
```

*Note: Include the dataset and PDF only if they are available and safe to share.*

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd mental-health-support-chatbot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot

```bash
python -m streamlit run main.py
```

Streamlit will provide a local URL to open the chatbot in your browser.

## Fine-Tuning

This project includes a LoRA-based fine-tuning implementation using Hugging Face Transformers and PEFT.

To start training, set `TRAIN_MODEL = True` in `main.py` and provide a compatible dataset in JSONL format.

After successful training, the saved adapter can be loaded into the chatbot. Training depends on model compatibility, dataset quality, and available hardware.

## Disclaimer

This chatbot is intended for educational and informational purposes only. It does not replace professional mental health care, diagnosis, or treatment.

## Author

Immama Shafique
