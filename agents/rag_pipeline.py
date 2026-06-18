import torch
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import config
import logging

# Set up logging for your demo visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_embeddings = None
_vectorstores = {}
_llm = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Force CPU usage to prevent 'meta-tensor' errors on local Windows/Mac setups
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu'}
        )
    return _embeddings

def get_vectorstore(domain: str = 'general'):
    global _vectorstores
    if domain not in _vectorstores:
        path = config.VECTORSTORE_PATHS.get(domain, config.VECTORSTORE_PATHS['general'])
        _vectorstores[domain] = FAISS.load_local(
            folder_path=path, 
            embeddings=get_embeddings(), 
            allow_dangerous_deserialization=True
        )
    return _vectorstores[domain]

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.LLM_MODEL)
    return _llm

PROMPT = ChatPromptTemplate.from_template('''
You are an expert Sri Lankan Insurance Advisor.
1. SCOPE: Only answer insurance-related questions. If the question is not about insurance, state: "I am a specialized Insurance Advisor and can only assist with inquiries related to Sri Lankan insurance policies."
2. CITATION: Cite source document names for all claims.
3. STRUCTURE: First sentence = direct answer. Second = LKR amount/detail. Follow with support.
4. COMPLIANCE: Never invent data.

DOMAIN: {domain}
LOCAL CONTEXT: {local_context}
USER PROFILE: {user_profile}
QUESTION: {question}

Answer with source citations:''')

def rag_query(question: str, domain: str = 'general', user_profile: dict = None) -> dict:
    logger.info(f"Retrieving from domain: {domain} | Question: {question}")
    
    try:
        retriever = get_vectorstore(domain).as_retriever(search_kwargs={'k': config.TOP_K_DOCS})
        
        # Profile Injection
        profile_str = f"Age: {user_profile.get('age', 'N/A')}, Job: {user_profile.get('job', 'N/A')}" if user_profile else "Not provided."
        hint = {'Retired': 'pension agrahara', 'Government Employee': 'agrahara nitf', 'Student': 'suraksha'}.get(user_profile.get('job', ''), '') if user_profile else ''
        
        local_docs = retriever.invoke(f"{question} {hint}".strip())
        context = '\n\n'.join([f'[SOURCE: {d.metadata.get("source","Doc")}]\n{d.page_content}' for d in local_docs])
        
        chain = PROMPT | get_llm()
        response = chain.invoke({'domain': domain.upper(), 'local_context': context, 'user_profile': profile_str, 'question': question})
        
        return {
            'answer': response.content, 
            'sources': list(set([d.metadata.get('source','?') for d in local_docs])), 
            'domain': domain
        }
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return {'answer': "I am currently unable to access the database. Please verify your vectorstore paths.", 'sources': [], 'domain': domain}