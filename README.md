# 📚 Educational RAG System

An AI-powered Retrieval-Augmented Generation (RAG) system designed for educational purposes. This system allows students to ask questions about lecture slides and receive detailed answers with source citations, as well as get personalized feedback on their own answers.

## ✨ Features

- **📄 PDF Processing**: Extract and index text from PDF lecture slides
- **💬 Question Answering**: Ask questions about lecture content and get detailed, cited answers
- **📝 Answer Feedback**: Submit your own answers and receive constructive feedback with scoring
- **🔍 Semantic Search**: Uses vector embeddings to find the most relevant slides for any question
- **📚 Source Citations**: Every answer includes references to specific slides and pages
- **🌐 Web Interface**: Beautiful Streamlit-based UI for easy interaction
- **💻 CLI Support**: Command-line interface for testing and automation

## 🏗️ Architecture

The system uses a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Processing**: PDFs are split into pages (slides) and text is extracted
2. **Embedding Generation**: Each slide is converted to a vector embedding using `sentence-transformers/all-MiniLM-L6-v2`
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Retrieval**: User queries are embedded and matched against stored slides
5. **Generation**: Relevant context is passed to Flan-T5-base model to generate answers

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)
- PDF files containing lecture slides

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd nlp_rag_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it (Mac/Linux)
   source venv/bin/activate
   
   # Activate it (Windows)
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your PDF files:**
   - Place your PDF lecture slides in the `data/` folder
   - Example: `data/Lecture_15.pdf`

## 📖 Usage

### Web Interface (Streamlit)

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Follow these steps in the web interface:**
   - Click "🚀 Initialize RAG System" in the sidebar (this loads the models)
   - Click "📑 Index [PDF name]" to index your PDF files
   - Use the "💬 Ask Questions" tab to ask questions
   - Use the "📝 Practice & Feedback" tab to get feedback on your answers

### Command Line Interface

1. **Run the test script:**
   ```bash
   python test_rag.py
   ```

2. **Or use in your own Python code:**
   ```python
   from rag_system import EducationalRAG
   
   # Initialize the system
   rag = EducationalRAG()
   
   # Index a PDF
   rag.index_documents("data/Lecture_15.pdf")
   
   # Ask a question
   result = rag.answer_question("What are the components of LSTM?")
   print(result['answer'])
   print(result['sources'])
   
   # Get feedback on an answer
   feedback = rag.provide_feedback(
       question="Why is LSTM better than RNN?",
       student_answer="LSTM has gates that help with memory"
   )
   print(feedback['feedback'])
   ```

## 📁 Project Structure

```
nlp_rag_project/
├── data/                  # Place your PDF files here
│   ├── Lecture_15.pdf
│   └── ...
├── app.py                 # Streamlit web application
├── rag_system.py          # Core RAG system implementation
├── test_rag.py           # CLI test script
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (created locally)
```

## 🔧 Technical Details

### Models Used

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Lightweight, CPU-friendly
  - 384-dimensional embeddings
  - Fast inference

- **Language Model**: `google/flan-t5-base`
  - Instruction-tuned for better responses
  - Good balance between quality and speed
  - Works well on CPU

- **Vector Database**: ChromaDB
  - In-memory storage
  - Fast similarity search
  - Easy to use Python API

### Key Methods

- `extract_text_from_pdf(pdf_path)`: Extracts text from PDF pages
- `index_documents(pdf_path)`: Processes and indexes PDF into vector database
- `retrieve_relevant_slides(query, top_k=5)`: Finds most relevant slides for a query
- `answer_question(question)`: Main Q&A function
- `provide_feedback(question, student_answer)`: Generates feedback on student answers

## 💡 Example Questions

- "What are the components of LSTM?"
- "Why is LSTM better than regular RNN?"
- "What is self-attention?"
- "Explain how transformers work"
- "What are the advantages of attention mechanisms?"

## 🎯 Use Cases

1. **Study Assistant**: Ask questions about lecture content to clarify concepts
2. **Practice Tool**: Submit your own answers and get feedback before exams
3. **Review System**: Quickly find relevant information from multiple lectures
4. **Learning Aid**: Understand complex topics with AI-generated explanations

## ⚠️ Notes

- **First Run**: The first time you run the system, it will download the models (~400MB). This may take a few minutes.
- **CPU Usage**: The system is designed to work on CPU, but GPU will be faster if available.
- **Memory**: Ensure you have at least 4GB of RAM available for model loading.
- **PDF Quality**: Text extraction works best with PDFs that have selectable text (not scanned images).

## 🔮 Future Improvements

- [ ] Support for multiple PDF formats (images, DOCX, etc.)
- [ ] Batch processing of multiple PDFs
- [ ] Conversation history and context
- [ ] Export answers and feedback to PDF
- [ ] Integration with more powerful LLMs (GPT, Claude, etc.)
- [ ] Multi-language support
- [ ] Advanced chunking strategies for better retrieval

## 📝 License

This project is part of a CS 52570 NLP course project at Purdue University Northwest.

## 👥 Credits

- Built with [Streamlit](https://streamlit.io/)
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/)
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
- Embeddings from [Sentence Transformers](https://www.sbert.net/)

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError` for PDF files
- **Solution**: Make sure your PDF files are in the `data/` folder and the filename matches exactly (including spaces/underscores)

**Issue**: Models not downloading
- **Solution**: Check your internet connection. Models are downloaded from Hugging Face on first use.

**Issue**: Out of memory errors
- **Solution**: The system uses Flan-T5-base which requires ~1.5GB RAM. Close other applications or use a smaller model.

**Issue**: Slow performance
- **Solution**: First-time model loading takes time. Subsequent runs will be faster. Consider using GPU if available.

---

**Happy Learning! 📚✨**
