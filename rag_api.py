from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import PromptTemplate
from duckduckgo_search import DDGS
import pandas as pd
import io
from typing import List, Dict, Any, Optional, AsyncIterator
import json
import time

class RAGService:
    def __init__(self, model_name: str = "mistral"):
        self.llm = Ollama(model=model_name, request_timeout=60.0)
        Settings.llm = self.llm
        self.index = None
        self.query_engine = None
        self.search_history = []
        
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search DuckDuckGo for real-time information"""
        try:
            print(f"Searching DuckDuckGo for: {query}")
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
                    print(f"Found result: {r.get('title', 'No title')}")
                    time.sleep(0.1)  # Small delay to avoid rate limiting
                
                self.search_history.append({"query": query, "results": results})
                print(f"Found {len(results)} results")
                return results
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
            return []
    
    def format_search_results(self, results: List[Dict]) -> str:
        """Format search results into a readable string"""
        if not results:
            return "No search results found."
            
        formatted = "Real-time information from online sources:\n\n"
        for i, result in enumerate(results):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            link = result.get('link', 'No link available')
            formatted += f"{i+1}. {title}\n{body}\nSource: {link}\n\n"
            
        return formatted

    def process_file(self, file_content: bytes, file_name: str) -> None:
        """Process a file and create an index"""
        file_extension = file_name.split('.')[-1].lower()
        
        try:
            if file_extension == "csv":
                df = pd.read_csv(io.BytesIO(file_content))
                docs = [Document(text=df.to_csv(index=False))]
            elif file_extension in ["xlsx", "xls"]:
                df = pd.read_excel(io.BytesIO(file_content))
                docs = [Document(text=df.to_csv(index=False))]
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Create index
            node_parser = MarkdownNodeParser()
            self.index = VectorStoreIndex.from_documents(
                documents=docs, 
                transformations=[node_parser], 
                show_progress=True
            )

            # Create query engine with streaming
            self.query_engine = self.index.as_query_engine(streaming=True)

            # Customize prompt template
            qa_prompt_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above, think step by step to answer the query in a highly precise and crisp manner focused on the final answer. If unsure, say 'I don't know!'.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
            self.query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    async def query(self, query: str) -> AsyncIterator[str]:
        """Query the RAG system with real-time data"""
        if not self.query_engine:
            raise Exception("No document has been processed yet")
            
        try:
            # Get real-time data from DuckDuckGo
            real_time_data = self.search_duckduckgo(query)
            real_time_context = self.format_search_results(real_time_data)
            
            # Combine with existing context
            enhanced_query = f"{query}\n\nReal-time context:\n{real_time_context}"
            
            streaming_response = self.query_engine.query(enhanced_query)
            async for chunk in streaming_response.response_gen:
                yield chunk
        except Exception as e:
            raise Exception(f"Error querying: {str(e)}")

    def get_context(self) -> Optional[str]:
        """Get the current context from the index"""
        if not self.index:
            return None
        return str(self.index.docstore.docs)
        
    def get_search_history(self) -> List[Dict]:
        """Get the search history"""
        return self.search_history 