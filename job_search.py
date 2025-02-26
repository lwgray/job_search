import os
import shutil
import streamlit as st
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from pathlib import Path


def process_document(file_path):
    """
    Process a document and add metadata based on its locatiion and type
    """
    category = ''
    if 'cover_letters' in str(file_path):
        category = 'cover_letter'
        
    elif 'resumes' in str(file_path):
        category = 'resume'
    elif 'descriptions' in str(file_path):
        category = 'job_description'
    
    def extract_role(file_path):
        """
        Extract the role from the file path
        """
        try:
            return file_path.stem.split('-')[0]
        except Exception as e:
            print(e)
            raise ValueError(f"Could not extract role from {file_path.stem}")
    
    def extract_company(file_path):
        """
        Extract the company from the file path
        """
        try:
            return file_path.stem.split('-')[1]
        except Exception as e:
            print(e)
            raise ValueError(f"Could not extract company from {file_path.stem}")

    role = extract_role(file_path)
    company = extract_company(file_path)

    try:
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()

         # If we got an empty document or very short content, print warning
        if not docs or (docs and len(docs[0].page_content.strip()) < 50):
            print(f"Warning: {file_path} yielded little or no content")
        
        for doc in docs:
            # Ensure we have meaninngful content
            if len(doc.page_content.strip()) < 10:
                continue
            
            doc.page_content = doc.page_content.replace('\u0000', '') # remove null characters
            doc.page_content = '\n'.join([line for line in doc.page_content.split('\n') if line.strip()]) # Remove empty lines

            # Add role and company to the top of the document
            header = f"Role: {role}\nCompany: {company}\n\n"
            doc.page_content = header + doc.page_content

            doc.metadata = {
                'source': str(file_path),
                'category': category,
                'role': role,                               
                'company': company,
                'filename': file_path.name
            }
        return docs
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return []

def load_documents():
    path = Path('./')
    docs = []
    for file in path.rglob('*.docx'):
        docs.extend(process_document(file))
    return docs

def initialize_embeddings():
    """
    Initialize the embeddings model
    """
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def create_vectorstore(chunks, embeddings):
    """
    Create a vector store from the chunks
    """
    if os.path.exists('.faiss_index'):
        shutil.rmtree('.faiss_index')
        print("Deleted existing FAISS index")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local('.faiss_index')
    return vectorstore

def setup_qa_chain(vectorstore):
    """
    Set up the QA chain with the retriever and LLM
    """
    k_value = len(list(Path('./descriptions').glob('*.docx')))

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={
            "k": k_value,
            "fetch_k": k_value * 3,
            "lambda_mult": 0.7,
            "filter": None
            }
        )
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022",
                        temperature=0,
                        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

def initialize_qa_chain():
    """
    Initialize the QA chain
    """
    try:
        st.info("Loading documents...")
        documents = load_documents()
        
        st.info("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", '\n', ". ", " ", "", ".  "],
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        st.info("Initializing HuggingFace Embeddings...")
        embeddings = initialize_embeddings()
        
        st.info("Creating vector store...")
        vectorstore = create_vectorstore(chunks, embeddings)

        st.info("Setting up QA chain...")
        qa_chain = setup_qa_chain(vectorstore)
            
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing the system: {e}")
        return None
    

def analyze_job_requirements():
    """
    Analyze job requirements across all job descriptions
    """
    question = """Analyze the job descriptions and list the most common:
    1. Technical skills required
    2. Years of experience needed
    3. Education requirements
    4. Soft skills mentioned"""
    
    result = st.session_state.qa_chain.invoke({"query": question})
    st.write(result["result"])
    
    with st.expander("View Source Documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.write(f"Source {i + 1}:")
            st.write(doc.page_content)
            st.write("---")

def generate_stats():
    """
    Generate statistics about the job search
    """
    stats_questions = [
        "How many unique companies am I applying to?",
        "What are the most common roles I'm applying for?",
        "What is the distribution of seniority levels in the jobs?",
        "What are the most common technical skills mentioned across all documents?"
    ]
    
    for question in stats_questions:
        st.subheader(question)
        with st.spinner("Analyzing..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            st.write(result["result"])
        st.write("---")

def add_analysis_features():
    """
    Add analysis feature buttons to sidebar and handle the display logic
    """
    st.sidebar.header("Analysis Tools")
    
    # Use radio buttons instead of individual buttons
    analysis_choice = st.sidebar.radio(
        "Select Analysis Tool",
        ["None", "Job Requirements Analysis", "Resume Comparison", "Application Stats"]
    )
    
    if analysis_choice == "Job Requirements Analysis":
        analyze_job_requirements()
    elif analysis_choice == "Resume Comparison":
        st.subheader("Resume Comparison Tool")
        job_descriptions = [f.name for f in Path('./descriptions').glob('*.docx')]
        resumes = [f.name for f in Path('./resumes').glob('*.docx')]
        
        selected_job = st.selectbox("Select Job Description", job_descriptions)
        selected_resume = st.selectbox("Select Resume", resumes)
        
        if st.button("Compare Documents"):
            question = f"""Compare the resume {selected_resume} with the job description {selected_job} and identify:
            1. Matching skills and qualifications
            2. Missing requirements
            3. Suggested improvements for the resume to better match this role
            4. Overall match percentage"""
            
            with st.spinner("Analyzing..."):
                result = st.session_state.qa_chain.invoke({"query": question})
                st.write(result["result"])
    elif analysis_choice == "Application Stats":
        generate_stats()


def main():
    st.title("Enhanced Resume Analysis Assistant")
    st.write("A tool to analyze your job search documents and improve your applications.")
    
    # Initialize the QA chain
    if 'qa_chain' not in st.session_state:
        with st.spinner("Loading documents and initializing the system..."):
            st.session_state.qa_chain = initialize_qa_chain()
            if st.session_state.qa_chain is None:
                st.error("Failed to initialize the system. Please check the logs and try again.")
                return
    
    # Add analysis features to sidebar
    add_analysis_features()
    
    # Main query interface
    st.header("Ask Questions")

    # Add document type filter
    doc_type = st.selectbox(
        "Filter by document type:",
        ["All Documents", "Cover Letters", "Resumes", "Job Descriptions"]
    )

    metadata_filter = None
    if doc_type != "All Documents":
        category_map = {
            "Resumes": "resume",    
            "Cover Letters": "cover_letter",
            "Job Descriptions": "job_description"
        }
        metadata_filter = {"category": category_map[doc_type]}

    user_question = st.text_input(
        "Enter your question:",
        "What are the most common skills mentioned across all job descriptions?"
    )
    
    if st.button("Ask"):
        with st.spinner("Analyzing..."):
            try:
                if metadata_filter:
                    retriever = st.session_state.qa_chain.retriever
                    filtered_docs = retriever.get_relevant_documents(
                        user_question,
                        filter=metadata_filter
                    )
                    result = st.session_state.qa_chain.invoke({
                        "query": user_question,
                        "input_documents": filtered_docs
                    })
                else:
                    result = st.session_state.qa_chain.invoke({"query": user_question})
                
                st.subheader("Answer:")
                st.write(result["result"])
                
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"Source {i + 1} ({doc.metadata.get('category', 'unknown')}):")
                        st.write(f"Role: {doc.metadata.get('role', 'unknown')}")
                        st.write(f"Company: {doc.metadata.get('company', 'unknown')}")
                        st.write(f"Filename: {doc.metadata.get('filename', 'unknown')}")
                        st.write("Content:")
                        st.write(doc.page_content)
                        st.write("---")
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

if __name__ == "__main__":
    main()
