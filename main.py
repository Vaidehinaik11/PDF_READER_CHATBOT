import os
from typing import Any, List, Optional, Dict
from dotenv import load_dotenv
import PyPDF2
import chromadb
from pydantic import BaseModel
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langgraph.graph import StateGraph, END

from prompt import comparator_prompt, timeline_prompt, aggregator_prompt, query_router_prompt

load_dotenv()

# fetching openAI api key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

class GraphState(BaseModel):
    """
    State model for the graph.
    """
    query: str
    query_type: Optional[str] = None
    comparator_answer: Optional[str] = None
    timeline_answer: Optional[str] = None
    aggregator_answer: Optional[str] = None

    comparator_docs: Optional[List[Dict[str, Any]]] = None
    timeline_docs: Optional[List[Dict[str, Any]]] = None
    aggregator_docs: Optional[List[Dict[str, Any]]] = None

retriever = None

def extract_pdf(files: List[str]) -> List[str]:
    """
    Extract text from a list of PDF files and split into chunks.
    """
    text = ''
    for file in files:
        read = PyPDF2.PdfReader(file)
        for i in read.pages:
            text += (i.extract_text()) 

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap= 100)
    chunk = splitter.split_text(text)
    return chunk

def create_retriever_with_chromadb(documents: List[str], model_name: str):
    """
    Create a Chroma retriever with HuggingFace embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    client = chromadb.PersistentClient(path="chromadb")
    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        collection_name="lang_docs",
        client=client,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

def run_rag_query(query: str, retriever, prompt_template: str) -> Dict[str, Any]:
    """
    Run a RAG query using the  MultiQueryRetriever and prompt template.

    returns:
               answer : str
               top_docs : list of document retrieved
               documents : metadata of the answer generated
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if isinstance(retriever, MultiQueryRetriever):
        mq_retriever = retriever
    else:
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
        )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=mq_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True)

    result = qa.invoke(query)
    if isinstance(result, dict):
        answer_text = result.get("result") or result.get("output_text") or ""
        source_docs = result.get("source_documents", [])
    else:
        answer_text = str(result)
        source_docs = []
    
    enriched_docs = []
    for i, doc in enumerate(source_docs):
        meta = doc.metadata or {}
        document_name = (
            meta.get("source")
            or meta.get("file_name")
            or meta.get("filename")
            or "Unknown"
        )
        page_number = meta.get("page") or meta.get("page_number")
        chunk_id = meta.get("chunk_id") or meta.get("id") or i
        score = meta.get("score") or meta.get("relevance_score")
        content = doc.page_content or ""
        max_len = 250
        excerpt = (content[:max_len] + "...") if len(content) > max_len else content

        enriched_docs.append({
            "document_name": document_name,
            "page_number": page_number,
            "chunk_id": chunk_id,
            "score": score,
            "excerpt": excerpt,
        })

    return {
        "answer": answer_text,
        "documents": enriched_docs,   
        "top_docs": source_docs,      
    }

def query_router(state: GraphState) -> GraphState:
    """
    Route the query to the appropriate processing node based on its type.

    returns:
        updated state with query_type set to 'comparator', 'timeline', or 'aggregator'
    """
    user_msg = f"Query: {state.query}"
    
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    resp = router_llm.invoke(
        [
            {"role": "system", "content": query_router_prompt},
            {"role": "user", "content": user_msg},
        ]
    )
    label = resp.content.strip().lower()

    if "timeline" in label:
        query_type = "timeline"
    elif "compar" in label:
        query_type = "comparator"
    elif "aggreg" in label:
        query_type = "aggregator"
    else:
        query_type = "aggregator"

    return state.model_copy(update={"query_type": query_type})

def route_from_query(state: GraphState) -> str:
    """
    Determine the next node based on the query_type in the state.
        returns:
        the name of the next node to route.
    """
    query_type = state.query_type

    if query_type == "comparator":
        return "comparator_node"
    elif query_type == "timeline":
        return "timeline_node"
    elif query_type == "aggregator":
        return "aggregator_node"
    else:
        return "aggregator_node"
    
def comparator_node(state: GraphState) -> GraphState:
    """
    Process the query as a comparator. it compares and constrast the information retrieved based on query.
        returns:
        updated state with comparator_answer
    """
    retriever = st.session_state.get("retriever")
    if retriever is None:
        raise ValueError("Retriever is not initialized. Upload PDFs and click submit first.")
    res = run_rag_query(state.query, retriever, comparator_prompt)
    return state.model_copy(update={"comparator_answer": res["answer"], "comparator_docs": res["documents"]})

def timeline_node(state: GraphState) -> GraphState:
    """
    Process the query as a timeline. it creates a timeline based on the information retrieved.
        returns:
        updated state with timeline_answer
    """
    retriever = st.session_state.get("retriever")
    if retriever is None:
        raise ValueError("Retriever is not initialized. Upload PDFs and click submit first.")
    res = run_rag_query(state.query, retriever, timeline_prompt)
    return state.model_copy(update={"timeline_answer": res["answer"], "timeline_docs": res["documents"]})

def aggregator_node(state: GraphState) -> GraphState:
    """
    Process the query as an aggregator. it aggregates the information retrieved to answer the query.
    returns:
        updated state with aggregator_answer and aggregator_docs
    """
    retriever = st.session_state.get("retriever")
    if retriever is None:
        raise ValueError("Retriever is not initialized. Upload PDFs and click submit first.")
    res = run_rag_query(state.query, retriever, aggregator_prompt)
    return state.model_copy(update={"aggregator_answer": res["answer"], "aggregator_docs": res["documents"]})


def main():
    """
    Define the state graph for routing queries based on their type. 
    the grapg consist of router node and three processing nodes: comparator, timeline, and aggregator.
    1. The router node uses an LLM to classify the query type.
    2. Based on the classification, the graph routes the query to the appropriate processing node.
    3. Each processing node executes a RAG query with a specific prompt template.
    4. The graph ends after processing the query in the selected node.
    """
    # define state graph
    graph = StateGraph(GraphState)
    
    # defining graph nodes
    graph.add_node("query_router", query_router)
    graph.add_node("comparator_node", comparator_node)
    graph.add_node("timeline_node", timeline_node)
    graph.add_node("aggregator_node", aggregator_node)

    # defining graph edges
    graph.set_entry_point("query_router")
    graph.add_conditional_edges("query_router", route_from_query)
    graph.add_edge("comparator_node", END)
    graph.add_edge("timeline_node", END)
    graph.add_edge("aggregator_node", END)

    # compiling the graph
    app = graph.compile()

    # vectors are store in steamlit session after its generated.
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    # session state is defined to save the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Streamlit UI started which consists of header, sidebar with pdf uploader and submit button and chat area with text input.
    st.header('PDF Analyser system')
    with st.sidebar:
        st.header("Menu:")
        pdf = st.file_uploader('upload pdf files:', accept_multiple_files= True, key = 'files')
        if st.button('submit', key= 'submit button'):
            if not pdf:
                st.warning("Please upload at least one PDF before submitting.")
            else:
                with st.spinner('processing........'):
                    text = extract_pdf(pdf)
                    embeding_model = "sentence-transformers/all-MiniLM-L6-v2"
                    retriever = create_retriever_with_chromadb(text, embeding_model)
                    st.session_state["retriever"] = retriever
                    st.success('done.')
    
        
        if st.session_state["retriever"] is not None:
            st.success("Retriever is ready. You can ask questions below.")
        else:
            st.info("Upload PDFs and click Submit to initialize the retriever.")

      
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View supporting documents"):
                    for i, d in enumerate(message["sources"], start=1):
                        document_name = d.get("document_name", "Unknown")
                        page_number = d.get("page_number", "N/A")
                        chunk_id = d.get("chunk_id", "N/A")
                        score = d.get("score", "N/A")
                        excerpt = d.get("excerpt", "")

                        st.markdown(f"**Source {i}**")
                        st.markdown(
                            f"- **Document:** {document_name}\n"
                            f"- **Page:** {page_number}\n"
                            f"- **Chunk ID:** `{chunk_id}`\n"
                            f"- **Score:** {score}"
                        )
                        if excerpt:
                            st.caption(excerpt)
                        st.markdown("---")
    
    # Streamlit chat input to get user queries and process them through the graph and generate final output.
    if prompt := st.chat_input("ask something"):
        with st.chat_message("user"):
                st.markdown(prompt)
        st.session_state.messages.append({"role":"user", "content": prompt})
        retriever = st.session_state.get("retriever")
        if retriever is None:
            st.warning("Please upload PDFs and click submit before asking a question.")
            return
        initial_state = GraphState(query=prompt)
        raw_state = app.invoke(initial_state)
        final_state = GraphState(**raw_state)

        if final_state.query_type == "comparator":
            response = final_state.comparator_answer
            docs = final_state.comparator_docs
        elif final_state.query_type == "timeline":
            response = final_state.timeline_answer
            docs = final_state.timeline_docs
        else:
            response = final_state.aggregator_answer
            docs = final_state.aggregator_docs

        with st.chat_message("assistant"):
            st.markdown(response)  

            if docs:
                with st.expander("View supporting documents"):
                    for i, d in enumerate(docs, start=1):
                        document_name = d.get("document_name", "Unknown")
                        page_number = d.get("page_number", "N/A")
                        chunk_id = d.get("chunk_id", "N/A")
                        score = d.get("score", "N/A")
                        excerpt = d.get("excerpt", "")

                        st.markdown(f"**Source {i}**")
                        st.markdown(
                            f"- **Document:** {document_name}\n"
                            f"- **Page:** {page_number}\n"
                            f"- **Chunk ID:** `{chunk_id}`\n"
                            f"- **Score:** {score}"
                        )
                        if excerpt:
                            st.caption(excerpt)
                        st.markdown("---")

        st.session_state.messages.append({"role":"assistant", "content": response, "sources": docs}) 

        print("\nQuery type decided by router:", final_state.query_type)
        print("\nComparator answer:\n", final_state.comparator_answer)
        print("\nTimeline answer:\n", final_state.timeline_answer)
        print("Answer:", final_state.aggregator_answer)
    

if __name__ == "__main__":
    main()
