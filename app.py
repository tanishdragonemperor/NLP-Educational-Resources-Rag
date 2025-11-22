import streamlit as st
from rag_system import EducationalRAG
import os

# Page config
st.set_page_config(
    page_title="Educational RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.indexed_files = []

# Title
st.title("📚 Educational RAG System")
st.markdown("*AI-Powered Learning Assistant with Citation & Feedback*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    # Check if data folder exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # List existing PDFs
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    
    if pdf_files:
        st.success(f"✅ Found {len(pdf_files)} PDF(s) in data folder")
        for pdf in pdf_files:
            st.write(f"📄 {pdf}")
    else:
        st.warning("⚠️ No PDFs found in /data folder")
    
    st.markdown("---")
    
    # Initialize RAG button
    if st.button("🚀 Initialize RAG System", type="primary"):
        with st.spinner("Loading models... This may take 1-2 minutes..."):
            st.session_state.rag_system = EducationalRAG()
        st.success("✅ RAG System Loaded!")
    
    # Index documents button
    if st.session_state.rag_system is not None and pdf_files:
        st.markdown("---")
        st.subheader("Index Documents")
        
        for pdf in pdf_files:
            if pdf not in st.session_state.indexed_files:
                if st.button(f"📑 Index {pdf}"):
                    with st.spinner(f"Indexing {pdf}..."):
                        pdf_path = os.path.join("data", pdf)
                        num_slides = st.session_state.rag_system.index_documents(pdf_path)
                        st.session_state.indexed_files.append(pdf)
                    st.success(f"✅ Indexed {num_slides} slides from {pdf}")
                    st.rerun()
        
        if st.session_state.indexed_files:
            st.markdown("**Indexed Documents:**")
            for indexed in st.session_state.indexed_files:
                st.write(f"✓ {indexed}")
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.write(f"**Model:** Flan-T5-Small")
    st.write(f"**Embeddings:** MiniLM-L6-v2")
    st.write(f"**Vector DB:** ChromaDB")

# Main content area
if st.session_state.rag_system is None:
    st.info("👈 Click **'Initialize RAG System'** in the sidebar to get started!")
    
    st.markdown("## 🎯 Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💬 Ask Questions
        - Get answers from your lecture slides
        - See exact source citations
        - Understand difficult concepts
        """)
    
    with col2:
        st.markdown("""
        ### 📝 Get Feedback
        - Submit your own answers
        - Receive personalized feedback
        - Improve your understanding
        """)

elif not st.session_state.indexed_files:
    st.warning("⚠️ Please index at least one PDF document from the sidebar before using the system.")

else:
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📝 Practice & Feedback"])
    
    # TAB 1: Q&A
    with tab1:
        st.header("💬 Ask Questions About Your Lectures")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the components of LSTM?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Get Answer", type="primary")
        
        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_system.answer_question(question)
            
            # Display answer
            st.markdown("### 📝 Answer")
            st.success(result['answer'])
            
            # Display sources
            st.markdown("### 📚 Sources")
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"📄 Source {i}: {source['source']} - Slide {source['page']}"):
                    st.text(source['text'][:500] + "...")
        
        elif ask_button:
            st.warning("⚠️ Please enter a question first!")
        
        # Example questions
        st.markdown("---")
        st.markdown("**💡 Example Questions:**")
        example_cols = st.columns(3)
        
        examples = [
            "What are the components of LSTM?",
            "Why is LSTM better than regular RNN?",
            "What is self-attention?"
        ]
        
        for col, example in zip(example_cols, examples):
            with col:
                if st.button(example, key=f"example_{example}"):
                    st.session_state.example_question = example
                    st.rerun()
        
        # Auto-fill example if clicked
        if 'example_question' in st.session_state:
            st.info(f"Try asking: {st.session_state.example_question}")
    
    # TAB 2: Practice & Feedback
    with tab2:
        st.header("📝 Practice & Get Feedback")
        
        st.markdown("""
        Submit your answer to a question and get personalized feedback!
        """)
        
        # Practice question input
        practice_question = st.text_input(
            "Question to practice:",
            placeholder="e.g., Explain how attention mechanism works"
        )
        
        # Student answer input
        student_answer = st.text_area(
            "Your Answer:",
            placeholder="Write your answer here...",
            height=150
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("📊 Get Feedback", type="primary")
        
        if submit_button and practice_question and student_answer:
            with st.spinner("🔍 Analyzing your answer..."):
                feedback_result = st.session_state.rag_system.provide_feedback(
                    practice_question,
                    student_answer
                )
            
            # Display feedback
            st.markdown("### 💬 Feedback")
            st.info(feedback_result['feedback'])
            
            # Display reference slides
            st.markdown("### 📚 Reference Material")
            for i, ref in enumerate(feedback_result['reference_slides'], 1):
                with st.expander(f"📄 Reference {i}: {ref['source']} - Slide {ref['page']}"):
                    st.text(ref['text'][:500] + "...")
        
        elif submit_button:
            if not practice_question:
                st.warning("⚠️ Please enter a question!")
            if not student_answer:
                st.warning("⚠️ Please write your answer!")
        
        # Example practice scenario
        st.markdown("---")
        st.markdown("**💡 Try This Example:**")
        
        if st.button("Load Example Practice Question"):
            st.session_state.example_practice = {
                'question': "Why is LSTM better than regular RNN?",
                'answer': "LSTM has gates that help with memory"
            }
            st.rerun()
        
        if 'example_practice' in st.session_state:
            st.info(f"""
            **Question:** {st.session_state.example_practice['question']}
            
            **Sample Answer:** {st.session_state.example_practice['answer']}
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📚 Educational RAG System | CS 52570 NLP Project | Purdue University Northwest</p>
</div>
""", unsafe_allow_html=True)