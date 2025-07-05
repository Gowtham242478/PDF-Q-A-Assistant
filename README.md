## 📘 PDF Question Answering Chatbot (Local GenAI App)

A Streamlit-based GenAI app that allows users to upload a PDF and ask questions about its contents using a local HuggingFace model (Flan-T5-Base).
No API key required. Runs **100% locally**.


🧠 Features

* ✅ Upload any readable text-based PDF
* ✅ Extracts and chunks the content for context search
* ✅ Ask natural language questions about the PDF
* ✅ Powered by open-source models (`flan-t5-base`, `MiniLM-L6-v2`)
* ✅ No internet or OpenAI API key required after setup
* ✅ Clean, user-friendly Streamlit UI



🖼️ Demo Screenshot

![image](https://github.com/user-attachments/assets/f873915c-1b76-4403-b593-b61a3a366e33)





🚀 How to Run

 1. Clone the Repo

```bash
git clone https://github.com/SHAIK2209
cd pdf-chatbot
```

#### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On macOS/Linux
```

#### 3. Install Requirements

```bash
| Package                   | Purpose                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| `streamlit`               | Web interface to run the app in your browser                          |
| `PyMuPDF` (`fitz`)        | To extract readable text from PDF documents                           |
| `langchain`               | Framework for chaining embeddings, vector search, and language models |
| `faiss-cpu`               | Vector store to store and search PDF text chunks efficiently          |
| `transformers`            | Loads local HuggingFace models like `flan-t5-base`                    |
| `sentence-transformers`   | Provides powerful embedding models like `MiniLM-L6-v2` for similarity |
| `torch`                   | Backend required to run models from `transformers`                    |
| `accelerate` *(optional)* | Speeds up model loading, safe to include                              |

```

#### 4. Run the App

```bash
streamlit run app.py
```

---

### 📂 Project Structure

```
pdf-chatbot/
├── app.py                # Streamlit user interface
├── qa_engine.py          # Logic: text splitting, embedding, Q&A
├── pdf_reader.py         # Extracts text from PDF using PyMuPDF
├── requirements.txt      # Python dependencies
└── README.md             # You’re reading it
```

---

### 📦 Dependencies

* Streamlit
* transformers
* langchain
* faiss-cpu
* PyMuPDF
* sentence-transformers

---

### 📄 Sample Questions You Can Ask

Once your PDF is uploaded, try:

* “What is a virtual machine?”
* “Explain demand paging.”
* “List the types of file access methods.”

---

### 👨‍💻 Author

I.V.N.R.S Gowtham – [GitHub(https://github.com/Gowtham242478)]
