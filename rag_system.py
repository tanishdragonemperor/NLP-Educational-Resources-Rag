import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

class EducationalRAG:
    def __init__(self):
        print("🔧 Initializing RAG System...")
        
        # Initialize embedding model (lightweight, CPU-friendly)
        print("📚 Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        print("💾 Setting up vector database...")
        self.chroma_client = chromadb.Client()
        try:
            self.chroma_client.delete_collection("lecture_slides")
        except:
            pass
        self.collection = self.chroma_client.create_collection("lecture_slides")
        
        # Initialize LLM (Flan-T5-BASE for better quality)
        print("🤖 Loading language model (this may take 2-3 minutes)...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        print("✅ RAG System Ready!\n")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with better chunking"""
        print(f"📄 Reading PDF: {pdf_path}")
        
        slides = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    # Clean up text
                    text = text.replace('\n', ' ').strip()
                    
                    slides.append({
                        'page_num': page_num + 1,
                        'text': text,
                        'source': os.path.basename(pdf_path)
                    })
        
        print(f"✅ Extracted {len(slides)} slides from PDF\n")
        return slides
    
    def index_documents(self, pdf_path):
        """Process PDF and add to vector database"""
        slides = self.extract_text_from_pdf(pdf_path)
        
        print("🔍 Creating embeddings and indexing...")
        for slide in slides:
            # Create embedding
            embedding = self.embedding_model.encode(slide['text']).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[slide['text']],
                metadatas=[{
                    'page': slide['page_num'],
                    'source': slide['source']
                }],
                ids=[f"{slide['source']}_page_{slide['page_num']}"]
            )
        
        print(f"✅ Indexed {len(slides)} slides into vector database\n")
        return len(slides)
    
    def retrieve_relevant_slides(self, query, top_k=5):
        """Find most relevant slides for a query - increased to 5 for better context"""
        # Create query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        relevant_slides = []
        for i in range(len(results['documents'][0])):
            relevant_slides.append({
                'text': results['documents'][0][i],
                'page': results['metadatas'][0][i]['page'],
                'source': results['metadatas'][0][i]['source']
            })
        
        return relevant_slides
    
    def generate_answer(self, query, context):
        """Generate answer using Flan-T5 with improved prompting"""
        # IMPROVED PROMPT - More specific instructions
        prompt = f"""You are an expert teaching assistant. Answer the student's question using ONLY the information from the lecture slides provided below. 

IMPORTANT INSTRUCTIONS:
- Provide a detailed, comprehensive answer (at least 3-4 sentences)
- Use specific information and examples from the slides
- Explain concepts clearly as if teaching a student
- If the slides mention specific components, gates, or mechanisms, explain them
- Do NOT make up information not in the slides

LECTURE SLIDES CONTENT:
{context}

STUDENT QUESTION: {query}

DETAILED ANSWER (3-4 sentences minimum):"""
        
        # Tokenize with increased length
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=1024,  # Increased from 512
            truncation=True
        )
        
        # Generate with better parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=300,      # Increased from 150
                min_length=50,       # NEW: Force longer answers
                num_beams=5,         # Increased from 4
                early_stopping=True,
                temperature=0.7,     # NEW: Add some creativity
                do_sample=False,     # Keep deterministic
                no_repeat_ngram_size=3  # Avoid repetition
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer_question(self, question):
        """Main Q&A function with better context assembly"""
        print(f"❓ Question: {question}")
        
        # Retrieve MORE relevant slides (5 instead of 3)
        relevant_slides = self.retrieve_relevant_slides(question, top_k=5)
        
        # Combine context with FULL text (not truncated)
        context_parts = []
        for i, slide in enumerate(relevant_slides, 1):
            context_parts.append(f"--- SLIDE {slide['page']} ---\n{slide['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Return top 3 slides for citation
        return {
            'answer': answer,
            'sources': relevant_slides[:3]  # Show top 3 for citations
        }
    
    def provide_feedback(self, question, student_answer):
        """Provide feedback on student's answer"""
        print(f"📝 Evaluating student answer...")
        
        # Get correct information from slides
        relevant_slides = self.retrieve_relevant_slides(question, top_k=3)
        
        # Use FULL text for reference
        correct_info = "\n\n".join([f"Slide {slide['page']}: {slide['text']}" 
                                    for slide in relevant_slides])
        
        # IMPROVED feedback prompt
        prompt = f"""You are a helpful teaching assistant providing detailed feedback to a student.

QUESTION: {question}

STUDENT'S ANSWER: 
{student_answer}

CORRECT INFORMATION FROM LECTURE SLIDES:
{correct_info}

Provide detailed, constructive feedback following this structure:

✅ CORRECT POINTS: What the student understood correctly (be specific)

❌ MISSING INFORMATION: What important points they missed from the slides (list specific concepts)

💡 IMPROVEMENT HINT: One specific suggestion to improve their answer

📊 SCORE: X/10 (with brief justification)

DETAILED FEEDBACK:"""
        
        # Generate feedback
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=350,      # Longer feedback
                min_length=80,       # Minimum length
                num_beams=5,
                early_stopping=True,
                temperature=0.7,
                no_repeat_ngram_size=3
            )
        
        feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'feedback': feedback,
            'reference_slides': relevant_slides
        }