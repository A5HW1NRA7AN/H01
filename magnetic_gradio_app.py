import gradio as gr
import asyncio
import json
import plotly.graph_objects as go
from magnetic_group_chat import TaskProcessor
import google.generativeai as genai
from duckduckgo_search import DDGS
import pandas as pd
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Configure Gemini
genai.configure(api_key="AIzaSyCKd5EVKcjmi780HyzuUaS_sx8EEXoB8fA")

# Set current date context
CURRENT_DATE = "April 8, 2025"

class SearchAgent:
    """Agent responsible for searching real-time information using DuckDuckGo"""
    
    def __init__(self):
        self.search_history = []
        
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search DuckDuckGo for real-time information"""
        try:
            print(f"Searching DuckDuckGo for: {query}")
            with DDGS() as ddgs:
                # Use the text search method with proper parameters
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

class HealthcareDemandPredictor:
    def __init__(self):
        self.task_processor = TaskProcessor()
        self.chat_history = []
        self.uploaded_files = []
        self.search_agent = SearchAgent()
        
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search DuckDuckGo for real-time information"""
        return self.search_agent.search(query, max_results)
        
    def process_files(self, files):
        """Process uploaded files and update chat context"""
        try:
            self.uploaded_files = files
            if not files:
                return "Please upload files to analyze."
                
            file_summaries = []
            for file in files:
                with open(file.name, 'rb') as f:
                    content = f.read()
                    result = asyncio.run(
                        self.task_processor.process_task(
                            f"Analyze the content of {file.name}",
                            content,
                            file.name
                        )
                    )
                    result_dict = json.loads(result)
                    file_summaries.append(result_dict["text"])
            
            summary = "\n\nFile Analysis Complete. You can now ask questions about the uploaded files."
            self.chat_history.append(["system", summary])
            return self.chat_history
            
        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            self.chat_history.append(["error", error_msg])
            return self.chat_history

    def chat_with_agents(self, message):
        """Process user message using Gemini-powered agents with real-time data"""
        try:
            # Add current date context to the message
            enhanced_message = f"{message} (Today's date is {CURRENT_DATE})"
            
            # First, let the search agent respond directly to the query
            search_query = f"healthcare demand {enhanced_message}"
            print(f"Starting search for: {search_query}")
            search_results = self.search_duckduckgo(search_query)
            
            # Format search results for context
            search_context = self.search_agent.format_search_results(search_results)
            
            # Add search results directly to chat history
            if search_results:
                self.chat_history.append(["search_agent", f"I found the following information about your query:\n\n{search_context}"])
            
            # Process the message with search context
            enhanced_message = f"{enhanced_message}\n\nContext from real-time sources:\n{search_context}"
            
            result = asyncio.run(
                self.task_processor.process_task(enhanced_message)
            )
            result_dict = json.loads(result)
            
            # Update chat history
            self.chat_history.append(["user", message])
            self.chat_history.append(["assistant", result_dict["text"]])
            
            # Add source links if available
            if search_results:
                source_links = "\n\nSources:\n"
                for i, result in enumerate(search_results):
                    link = result.get('link', 'No link available')
                    source_links += f"{i+1}. {link}\n"
                self.chat_history.append(["system", source_links])
                print(f"Added {len(search_results)} source links to chat history")
            
            # Create visualization if numeric data is available
            fig = None
            numeric_data = result_dict.get("numeric_data", {})
            
            if numeric_data.get("time_series"):
                fig = go.Figure()
                time_series = numeric_data["time_series"]
                fig.add_trace(go.Scatter(y=time_series, mode='lines+markers'))
                fig.update_layout(title="Time Series Analysis")
            
            elif numeric_data.get("comparison"):
                fig = go.Figure()
                comparison = numeric_data["comparison"]
                fig.add_trace(go.Bar(y=comparison))
                fig.update_layout(title="Comparison Analysis")
            
            return self.chat_history, fig if fig else None
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.chat_history.append(["error", error_msg])
            return self.chat_history, None

    def clear_chat(self):
        """Clear chat history"""
        self.chat_history = []
        return [], None

def create_interface():
    predictor = HealthcareDemandPredictor()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Healthcare Demand Prediction System")
        gr.Markdown(f"*Current date: {CURRENT_DATE}*")
        
        with gr.Row():
            with gr.Column():
                file_output = gr.File(
                    file_count="multiple",
                    label="Upload Files"
                )
                process_button = gr.Button("Process Files")
            
            with gr.Column():
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400
                )
                
                msg = gr.Textbox(
                    label="Enter your message",
                    placeholder="Ask about healthcare demand, SDoH factors, or resource allocation..."
                )
                
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear Chat")
                
                plot = gr.Plot(label="Visualization")
        
        # Event handlers
        process_button.click(
            fn=predictor.process_files,
            inputs=[file_output],
            outputs=[chatbot]
        )
        
        submit.click(
            fn=predictor.chat_with_agents,
            inputs=[msg],
            outputs=[chatbot, plot]
        )
        
        clear.click(
            fn=predictor.clear_chat,
            inputs=[],
            outputs=[chatbot, plot]
        )
        
        # Example prompts
        gr.Examples(
            examples=[
                ["What are the key healthcare demand trends in the uploaded data?"],
                ["Analyze the social determinants of health affecting resource allocation."],
                ["Generate a forecast for healthcare resource requirements."],
                ["What are the main risk factors identified in the analysis?"]
            ],
            inputs=msg
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()