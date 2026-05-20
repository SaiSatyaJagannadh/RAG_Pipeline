from typing import List, Tuple
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.docstore.document import Document

from langchain_core .globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from .utils import get_vector_store

from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever




SYSTEM = """You are a grounded company knowledge assistant.
Always base answers strictly on the provided context.
If the answer isn't present, reply with "I don't know."
Respond concisely and clearly.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Question:\n{input}\n\n"
     "Context:\n{context}\n\n"
     "Rule: Prefer the most recent policy by effective date.")
])

REDIS_URL = os.getenv("REDIS_URL")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


set_llm_cache(
    RedisSemanticCache(
        redis_url=REDIS_URL,
        embeddings =embeddings,
        distance_threshold = 0.98
    )
)



async def _build_chain():
    store = await get_vector_store()  
    
    
    
    
    
    retriever=store.as_retriever(search_kwargs={"k": int(os.getenv("RETRIEVAL_K","5"))})

    llm = ChatOpenAI(model="gpt-4o-mini")
    doc_chain = create_stuff_documents_chain(llm, PROMPT)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain

async def answer_with_docs_async(question: str) -> Tuple[str, List[str]]:
    
    chain = await _build_chain()
    result = await chain.ainvoke({"input": question})
    answer= result["answer"]

    sources=[]
    docs: List[Document] = result["context"]
    
    unique_sources = {d.metadata.get("source") for d in docs}
    sources = sorted(unique_sources)




    return answer, sources #rag,api,evals needs ro change 
