
import os
import streamlit as st
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configuration

MODEL_NAME = "thrishala/mental_health_chatbot"
DATA_PATH = "mental_health_data.jsonl"
ADAPTER_PATH = "./fine_tuned_model"
PDF_PATH = r"D:\gen_ai\mental_health.pdf"

# set True to train and false to run the chatbot
TRAIN_MODEL = False


# fine-tuning

def fine_tune_model():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Runtime Error"
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training file not found: {DATA_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    )

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256
        )

    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="./training_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        logging_steps=5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    trainer.train()

    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    print("Fine-tuning complete!")

# load chatbot

@st.cache_resource
def load_chatbot():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    if os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline=generator)


# PDF loading

@st.cache_resource
def load_faiss():

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF nahi mili: {PDF_PATH}"
        )

    documents = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/msmarco-distilbert-base-v4"
    )

    return FAISS.from_documents(chunks, embeddings)


# generate response

def generate_response(query, llm, vector_store, memory):

    docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = PromptTemplate(
        input_variables=["history", "context", "input"],
        template="""
You are a supportive informational chatbot.
Use the PDF context to answer the user's question.
If the answer is not in the context, say so.
Do not diagnose conditions or prescribe medication.

Conversation:
{history}

PDF context:
{context}

User question:
{input}

Answer:
"""
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    return chain.predict(
        context=context,
        input=query
    )


# Streamlit application

if TRAIN_MODEL:
    st.title("Fine-tune Mental Health Chatbot")

    if st.button("Start Fine-tuning"):
        with st.spinner("Training model..."):
            try:
                fine_tune_model()
                st.success("Fine-tuning complete!")
            except Exception as error:
                st.error(str(error))

else:
    st.title("Mental Health Support Chatbot")

    st.warning(
        "Educational support only, this chatbot does not provide diagnosis."
    )

    try:
        llm = load_chatbot()
        vector_store = load_faiss()

    except Exception as error:
        st.error(str(error))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            input_key="input"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        try:
            with st.spinner("Generating response..."):
                answer = generate_response(
                    query,
                    llm,
                    vector_store,
                    st.session_state.memory
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.write(answer)

        except Exception as error:
            st.error(f"Response error: {error}")
