from rag_system import EducationalRAG

rag = EducationalRAG()

rag.index_documents("data/Lecture 15.pdf")

print("\n" + "="*50)
print("TESTING Q&A")
print("="*50)

result = rag.answer_question("What are the components of LSTM?")
print(f"\n📝 Answer: {result['answer']}")
print(f"\n📚 Sources:")
for source in result['sources']:
    print(f"  - {source['source']}, Slide {source['page']}")

# Test Feedback
print("\n" + "="*50)
print("TESTING FEEDBACK")
print("="*50)

feedback = rag.provide_feedback(
    question="Why is LSTM better than RNN?",
    student_answer="LSTM has gates"
)
print(f"\n💬 Feedback:\n{feedback['feedback']}")