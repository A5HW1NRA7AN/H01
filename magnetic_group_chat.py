from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import List, Dict, Optional, AsyncIterator
from rag_api import RAGService
import json
import asyncio
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyCKd5EVKcjmi780HyzuUaS_sx8EEXoB8fA")
model = genai.GenerativeModel('gemini-2.0-flash')

class MagneticGroupChat:
    def __init__(self, model_client: OllamaChatCompletionClient):
        self.model_client = model_client
        self.rag_service = RAGService()
        self.orchestrator = self._create_orchestrator()
        self.agents = self._create_agents()
        self.message_history = []
        self.current_context = None
        
    def _create_orchestrator(self) -> AssistantAgent:
        """Creates the orchestrator (manager) agent"""
        return AssistantAgent(
            "orchestrator",
            model_client=self.model_client,
            description="Orchestrates the healthcare demand prediction workflow and manages the entire pipeline",
            system_message="""You are the central orchestrator that coordinates all agents in the healthcare demand prediction system.
            Your core responsibilities:
            1. Task Management
               - Create and maintain execution plans for healthcare demand analysis
               - Assign tasks to appropriate specialized agents
               - Track task progress and completion
               - Handle task dependencies between data collection, analysis, and visualization
            
            2. Performance Monitoring
               - Track agent performance metrics in healthcare data processing
               - Monitor system efficiency in handling medical data
               - Identify bottlenecks in the prediction pipeline
               - Report on task success rates and prediction accuracy
            
            3. Quality Control
               - Ensure task completion standards for healthcare analytics
               - Validate agent outputs for medical relevance
               - Maintain data consistency across healthcare metrics
               - Enforce quality thresholds for prediction reliability
            
            
            When a task is complete, respond with TERMINATE."""
        )
    
    def _create_agents(self) -> Dict[str, AssistantAgent]:
        """Creates the specialized agents for healthcare demand prediction"""
        agents = {
            "deployment_agent": AssistantAgent(
                "deployment_agent",
                model_client=self.model_client,
                description="Hosts web interface for healthcare data input/output",
                system_message="""You are the Deployment Agent responsible for managing user interactions through the healthcare demand prediction web interface.
                Your core responsibilities:
                1. Interface Management
                   - Handle user inputs and healthcare data file uploads
                   - Process medical and SDoH data files
                   - Validate input formats (CSV, Excel)
                   - Manage user sessions for healthcare analysis
                
                2. Data Validation
                   - Check SDoH data completeness and accuracy
                   - Validate healthcare metrics and indicators
                   - Ensure data format consistency for medical records
                   - Track data quality scores for healthcare information
                
                3. Response Handling
                   - Format healthcare demand prediction outputs
                   - Display medical data visualizations
                   - Provide error messages for healthcare data issues
                   - Track user feedback on prediction accuracy
                
                When complete, respond with TERMINATE."""
            ),
            
            "planner_agent": AssistantAgent(
                "planner_agent",
                model_client=self.model_client,
                description="Orchestrates healthcare demand prediction workflow and manages feedback",
                system_message="""You are the Planner Agent responsible for orchestrating the healthcare demand prediction workflow.
                Your core responsibilities:
                1. Pipeline Management
                   - Coordinate agent activities for healthcare analysis
                   - Manage data flow between medical data sources
                   - Handle dependencies in the prediction pipeline
                   - Track progress of healthcare demand forecasts
                
                2. Task Delegation
                   - Assign healthcare analysis tasks to agents
                   - Set priorities for medical data processing
                   - Monitor completion of prediction tasks
                   - Handle failures in healthcare data analysis
                
                3. Quality Control
                   - Review agent outputs for healthcare relevance
                   - Manage feedback loops for prediction accuracy
                   - Ensure consistency in medical data analysis
                   - Track performance of healthcare forecasting
                
                When complete, respond with TERMINATE."""
            ),
            
            "research_agent": AssistantAgent(
                "research_agent",
                model_client=self.model_client,
                description="Gathers real-time healthcare and SDoH data from online sources",
                system_message="""You are the Research Agent responsible for gathering real-time Social Determinants of Health (SDoH) data from online sources.
                Your core responsibilities:
                1. Data Collection
                   - Search and collect real-time SDoH metrics from government health databases
                   - Monitor healthcare indicators from CDC, WHO, and local health departments
                   - Track socioeconomic factors from Census Bureau and economic databases
                   - Gather demographic data from population health surveys
                
                2. Data Processing
                   - Clean and standardize collected healthcare data
                   - Convert text data to numerical metrics for analysis
                   - Calculate derived healthcare indicators
                   - Validate consistency of medical data
                
                3. Trend Analysis
                   - Identify patterns in healthcare utilization
                   - Calculate changes in demand for medical services
                   - Monitor thresholds for healthcare capacity
                   - Flag anomalies in patient volumes
                
                When performing searches:
                1. Use the search_duckduckgo function to find relevant information
                2. Format your findings in a structured JSON format
                3. Include source links for all information
                4. Highlight key metrics and trends
                
                Output Format:
                {
                    "search_query": "your search query",
                    "findings": [
                        {
                            "title": "finding title",
                            "description": "detailed description",
                            "source": "source URL",
                            "metrics": ["metric1", "metric2"],
                            "relevance_score": 0.95
                        }
                    ],
                    "summary": "brief summary of findings",
                    "trends": ["trend1", "trend2"],
                    "recommendations": ["recommendation1", "recommendation2"]
                }
                
                When complete, respond with TERMINATE."""
            ),
            
            "rag_agent": AssistantAgent(
                "rag_agent",
                model_client=self.model_client,
                description="Manages RAG storage/retrieval for healthcare historical and current context",
                system_message="""You are the RAG Agent responsible for managing the Retrieval-Augmented Generation system for healthcare data.
                Your core responsibilities:
                1. Context Management
                   - Store historical healthcare data (2011-2020)
                   - Index current medical information
                   - Maintain data relationships between health indicators
                   - Update context with latest healthcare trends
                
                2. Query Processing
                   - Retrieve relevant healthcare context
                   - Combine historical/current medical data
                   - Calculate relevance scores for health queries
                   - Filter results for medical accuracy
                
                3. Knowledge Integration
                   - Merge healthcare data sources
                   - Resolve conflicts in medical information
                   - Track data lineage for health metrics
                   - Maintain freshness of healthcare data
                
                When complete, respond with TERMINATE."""
            ),
            
            "risk_agent": AssistantAgent(
                "risk_agent",
                model_client=self.model_client,
                description="Generates healthcare demand forecasts and risk scores",
                system_message="""You are the Risk Agent responsible for healthcare demand forecasting and risk assessment.
                Your core responsibilities:
                1. Demand Forecasting
                   - Generate healthcare demand predictions
                   - Calculate confidence intervals for medical service needs
                   - Identify demand spikes for emergency services
                   - Track accuracy of healthcare forecasts
                
                2. Risk Assessment
                   - Calculate risk scores for healthcare capacity
                   - Identify risk factors affecting patient volumes
                   - Assess impact levels on medical resources
                   - Monitor thresholds for healthcare system stress
                
                3. Pattern Analysis
                   - Detect seasonality in healthcare utilization
                   - Identify trends in patient demographics
                   - Analyze correlations between SDoH and health outcomes
                   - Flag anomalies in healthcare demand patterns
                
                When complete, respond with TERMINATE."""
            ),
            
            "allocation_agent": AssistantAgent(
                "allocation_agent",
                model_client=self.model_client,
                description="Plans healthcare resource allocation based on demand forecasts",
                system_message="""You are the Allocation Agent responsible for healthcare resource planning.
                Your core responsibilities:
                1. Resource Planning
                   - Analyze current healthcare capacity
                   - Project medical resource needs
                   - Optimize allocation of healthcare staff and equipment
                   - Track utilization of medical facilities
                
                2. Efficiency Analysis
                   - Calculate efficiency metrics for healthcare delivery
                   - Identify bottlenecks in medical service provision
                   - Optimize distribution of healthcare resources
                   - Monitor costs of medical operations
                
                3. Implementation Strategy
                   - Create action plans for healthcare resource deployment
                   - Set priorities for medical service expansion
                   - Define timelines for healthcare capacity changes
                   - Track progress of medical resource allocation
                
                When complete, respond with TERMINATE."""
            ),
            
            "visualization_agent": AssistantAgent(
                "visualization_agent",
                model_client=self.model_client,
                description="Creates healthcare data visualizations from outputs",
                system_message="""You are the Visualization Agent responsible for creating clear and informative healthcare data visualizations.
                Your core responsibilities:
                1. Graph Generation
                   - Create healthcare demand trend plots
                   - Visualize medical resource allocation
                   - Show risk distributions for healthcare capacity
                   - Display SDoH correlations with health outcomes
                
                2. Interactive Features
                   - Add drill-down capabilities for healthcare metrics
                   - Enable filtering by medical specialties
                   - Support comparisons of healthcare regions
                   - Include tooltips with medical data details
                
                3. Layout Optimization
                   - Arrange multiple healthcare plots
                   - Ensure readability of medical data
                   - Maintain consistency in visualization style
                   - Support responsiveness for different devices
                
                When complete, respond with TERMINATE."""
            ),
            
            "critic_agent": AssistantAgent(
                "critic_agent",
                model_client=self.model_client,
                description="Reviews and refines healthcare outputs iteratively",
                system_message="""You are the Critic Agent responsible for quality assurance and refinement of healthcare predictions.
                Your core responsibilities:
                1. Output Review
                   - Validate healthcare demand predictions
                   - Check methodology for medical data analysis
                   - Assess completeness of health indicators
                   - Verify accuracy of healthcare forecasts
                
                2. Refinement
                   - Suggest improvements to healthcare models
                   - Identify gaps in medical data coverage
                   - Enhance clarity of healthcare visualizations
                   - Optimize formats for medical reporting
                
                3. Quality Metrics
                   - Track performance of healthcare predictions
                   - Monitor consistency of medical data analysis
                   - Measure reliability of health forecasts
                   - Report issues with healthcare capacity planning
                
                
                When complete, respond with TERMINATE."""
            )
        }
        return agents

    def create_group_chat(self) -> RoundRobinGroupChat:
        """Creates the group chat with all agents"""
        all_agents = [self.orchestrator] + list(self.agents.values())
        return RoundRobinGroupChat(
            participants=all_agents,
            termination_condition=TextMentionTermination("TERMINATE")
        )

    async def process_document(self, file_content: bytes, file_name: str) -> AsyncIterator[Dict]:
        """Process a document using RAG"""
        try:
            self.rag_service.process_file(file_content, file_name)
            self.current_context = self.rag_service.get_context()
            yield {"source": "rag_agent", "content": f"Successfully processed document: {file_name}"}
        except Exception as e:
            yield {"source": "rag_agent", "content": f"Error processing document: {str(e)}"}

    async def query_document(self, query: str) -> AsyncIterator[Dict]:
        """Query the RAG system"""
        try:
            async for chunk in self.rag_service.query(query):
                yield {"source": "rag_agent", "content": chunk}
        except Exception as e:
            yield {"source": "rag_agent", "content": f"Error querying document: {str(e)}"}

    async def process_task(self, task: str) -> AsyncIterator[Dict]:
        """Process a task through the agent system with RAG context"""
        group_chat = self.create_group_chat()
        
        # If we have RAG context, add it to the task
        if self.current_context:
            enhanced_task = f"""Task: {task}

Available Context from RAG:
-------------------------
{self.current_context}
-------------------------

Please use this context when relevant for your analysis and predictions."""
        else:
            enhanced_task = task

        # Add search history to the task if available
        search_history = self.rag_service.get_search_history()
        if search_history:
            search_context = "Recent search history:\n"
            for entry in search_history[-3:]:  # Include last 3 searches
                search_context += f"Query: {entry['query']}\n"
                search_context += f"Results: {len(entry['results'])} found\n\n"
            
            enhanced_task = f"{enhanced_task}\n\n{search_context}"

        async for message in group_chat.run_stream(task=enhanced_task):
            # Handle different message types
            if hasattr(message, 'source') and hasattr(message, 'content'):
                # Handle TextMessage objects
                source = message.source
                content = message.content
            elif hasattr(message, 'get'):
                # Handle dictionary-like objects
                source = message.get("source", "Unknown")
                content = message.get("content", "")
            else:
                # Handle TaskResult objects
                source = getattr(message, "source", "Unknown")
                content = getattr(message, "content", str(message))
                
            # If message is from research_agent, try to perform a search
            if source == "research_agent" and content:
                try:
                    # Extract potential search queries from the research agent's message
                    search_queries = [
                        f"healthcare demand prediction {content[:50]}",
                        f"social determinants of health {content[:50]}",
                        f"healthcare trends {content[:50]}"
                    ]
                    
                    for query in search_queries:
                        search_results = self.rag_service.search_duckduckgo(query, max_results=3)
                        if search_results:
                            formatted_results = self.rag_service.format_search_results(search_results)
                            yield {
                                "source": "research_agent",
                                "content": f"Search results for '{query}':\n{formatted_results}"
                            }
                except Exception as e:
                    print(f"Error in research agent search: {e}")
                    
            # If message is from rag_agent, try to get additional context
            if source == "rag_agent" and content:
                try:
                    async for rag_response in self.rag_service.query(content):
                        yield {
                            "source": "rag_agent",
                            "content": f"Additional context: {rag_response}"
                        }
                except Exception:
                    pass
                    
            yield {"source": source, "content": content}

class TaskProcessor:
    def __init__(self, model_name: str = "mistral"):
        self.model_client = OllamaChatCompletionClient(model=model_name)
        self.magnetic_chat = MagneticGroupChat(self.model_client)
        self.gemini_chat = model.start_chat(history=[])
    
    async def process_task(self, task: str, file_content: bytes = None, file_name: str = None) -> str:
        """Process a task using Gemini and return formatted results"""
        print(f"\nProcessing task: {task}\n")
        print("=" * 80)
        
        messages = []
        numeric_data = {
            "time_series": [],
            "comparison": []
        }
        
        try:
            # Process document if provided
            if file_content and file_name:
                # Update Gemini chat with file context
                file_info = f"Processing file: {file_name}"
                context_msg = self.gemini_chat.send_message(file_info)
                messages.append({
                    "source": "system",
                    "content": context_msg.text
                })
                
                # Process through RAG system
                async for message in self.magnetic_chat.process_document(file_content, file_name):
                    messages.append({
                        "source": message["source"],
                        "content": message["content"]
                    })
            
            # Prepare the prompt for Gemini
            prompt = f"""Task: {task}

Please analyze this healthcare-related task considering:
1. Social Determinants of Health (SDoH)
2. Healthcare Demand Patterns
3. Resource Requirements
4. Risk Factors

Provide a structured response with:
- Key findings and insights
- Relevant metrics and trends
- Specific recommendations
- Risk assessment
- Resource allocation suggestions

Format numeric data clearly for visualization."""

            # Get response from Gemini
            response = self.gemini_chat.send_message(prompt)
            
            # Extract potential numeric data
            try:
                # Look for time series data
                import re
                time_series_matches = re.findall(r'Time Series.*?:\s*\[(.*?)\]', response.text, re.DOTALL)
                if time_series_matches:
                    numeric_data["time_series"] = json.loads(f"[{time_series_matches[0]}]")
                
                # Look for comparison data
                comparison_matches = re.findall(r'Comparison.*?:\s*\[(.*?)\]', response.text, re.DOTALL)
                if comparison_matches:
                    numeric_data["comparison"] = json.loads(f"[{comparison_matches[0]}]")
            except:
                pass  # Continue if numeric data extraction fails
            
            messages.append({
                "source": "gemini",
                "content": response.text
            })
            
        except Exception as e:
            messages.append({
                "source": "error",
                "content": f"Error processing task: {str(e)}"
            })
        finally:
            await self.model_client.close()
        
        # Format the final result
        result = {
            "text": "\n".join([f"{m['source']}: {m['content']}" for m in messages]),
            "numeric_data": numeric_data,
            "message_history": messages
        }
        
        return json.dumps(result)

# Example usage
async def main():
    processor = TaskProcessor()
    
    while True:
        print("\n=== Healthcare Demand Prediction System ===")
        print("1. Process a new document")
        print("2. Run analysis task")
        print("3. Query existing data")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            file_path = input("\nEnter the path to your data file (CSV/Excel): ")
            # Remove quotes if present
            file_path = file_path.strip('"\'')
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                    await processor.process_task("Process and index the document", file_content, file_path)
                print("\nDocument processed successfully!")
            except Exception as e:
                print(f"\nError processing file: {e}")
        
        elif choice == "2":
            print("\nSelect analysis type:")
            print("1. Full demand prediction")
            print("2. Historical analysis")
            print("3. Resource planning")
            print("4. Custom analysis")
            
            analysis_choice = input("\nEnter analysis type (1-4): ")
            
            if analysis_choice == "1":
                task = """
                Perform comprehensive healthcare demand prediction:
                1. Analyze historical patterns
                2. Generate demand forecasts
                3. Plan resource allocation
                4. Identify risk factors
                """
            elif analysis_choice == "2":
                task = """
                Conduct historical healthcare data analysis:
                1. Review utilization trends
                2. Identify seasonal patterns
                3. Analyze SDoH correlations
                """
            elif analysis_choice == "3":
                task = """
                Create resource allocation plan:
                1. Assess current resources
                2. Project future needs
                3. Optimize distribution
                """
            else:
                task = input("\nEnter your custom analysis requirements: ")
            
            await processor.process_task(task)
        
        elif choice == "3":
            query = input("\nEnter your query about the processed data: ")
            async for message in processor.magnetic_chat.query_document(query):
                print(f"\n{message['source']}: {message['content']}")
        
        elif choice == "4":
            print("\nExiting system...")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem interrupted by user. Exiting...") 