from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, GroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import List, Dict
import json
import google.generativeai as genai

# Initialize the model clients
model_client = OllamaChatCompletionClient(model="mistral")
gemini_client = OpenAIChatCompletionClient(
    api_key="AIzaSyCKd5EVKcjmi780HyzuUaS_sx8EEXoB8fA",
    model="gemini-2.0-flash"
)

# Configure Gemini
genai.configure(api_key="AIzaSyCKd5EVKcjmi780HyzuUaS_sx8EEXoB8fA")

class DataMessage:
    def __init__(self, sender: str, data_type: str, content: dict):
        self.sender = sender
        self.data_type = data_type
        self.content = content
        self.timestamp = None  # Can be added if needed

    def to_json(self):
        return json.dumps({
            "sender": self.sender,
            "data_type": self.data_type,
            "content": self.content
        })

# Define the Manager Agent (Cone Apex)
manager_agent = AssistantAgent(
    "manager_agent",
    model_client=model_client,
    description="Coordinates all agent groups for healthcare demand prediction",
    system_message="""You are the Manager Agent responsible for coordinating all agent groups in the healthcare demand prediction system.
    Your core responsibilities:
    1. Task Coordination
       - Manage healthcare data processing tasks
       - Coordinate medical prediction workflows
       - Handle healthcare visualization requests
       - Track progress of medical analysis
    
    2. Data Flow Management
       - Route healthcare data between agents
       - Ensure medical data consistency
       - Monitor healthcare metrics flow
       - Validate medical data integrity
    
    3. Task Sequencing
       - Order healthcare analysis steps
       - Manage medical prediction dependencies
       - Track healthcare task completion
       - Handle medical data priorities
    
    Output Format (JSON):
    {
        "task_status": {
            "current_task": "healthcare_demand_forecast",
            "progress": 0.75,
            "next_steps": ["resource_allocation", "visualization"],
            "dependencies": ["sdoh_data", "historical_trends"]
        },
        "data_flow": {
            "source": "research_agent",
            "destination": "risk_agent",
            "data_type": "healthcare_metrics",
            "timestamp": "2024-03-15T10:30:00Z"
        }
    }
    
    When complete, respond with TERMINATE."""
)

# Data Processing Group
data_group_agents = [
    AssistantAgent(
        "research_agent",
        model_client=gemini_client,  # Using Gemini for research
        description="Collects and processes real-time healthcare and SDoH data",
        system_message="""You are the Research Agent responsible for collecting and processing real-time healthcare and SDoH data.
        Your core responsibilities:
        1. Data Collection
           - Gather healthcare metrics from government sources
           - Collect SDoH data from health databases
           - Monitor medical service utilization
           - Track healthcare capacity indicators
        
        2. Data Processing
           - Clean healthcare data
           - Standardize medical metrics
           - Calculate derived health indicators
           - Validate healthcare data quality
        
        3. Data Analysis
           - Identify healthcare trends
           - Analyze SDoH correlations
           - Calculate medical service demand
           - Monitor healthcare capacity
        
        Output Format (JSON):
        {
            "healthcare_data": {
                "demographics": {
                    "population": 500000,
                    "age_distribution": {"0-18": 0.25, "19-64": 0.60, "65+": 0.15},
                    "health_insurance": 0.92
                },
                "health_indicators": {
                    "chronic_conditions": 0.35,
                    "emergency_visits": 1200,
                    "hospital_admissions": 450
                },
                "sdoh_metrics": {
                    "income": 65000,
                    "education": 0.85,
                    "housing": 0.78
                }
            },
            "analysis_results": {
                "trends": ["increasing_demand", "aging_population"],
                "correlations": {"income_health": 0.75, "education_outcomes": 0.82},
                "capacity_pressure": 0.65
            }
        }
        
        When complete, respond with TERMINATE."""
    ),
    AssistantAgent(
        "rag_agent",
        model_client=model_client,
        description="Processes and indexes healthcare documents for context retrieval",
        system_message="""You are the RAG Agent responsible for processing and indexing healthcare documents.
        Your core responsibilities:
        1. Document Processing
           - Index healthcare records
           - Process medical guidelines
           - Store SDoH documentation
           - Update healthcare knowledge base
        
        2. Context Retrieval
           - Find relevant healthcare information
           - Match medical queries to documents
           - Rank healthcare context by relevance
           - Filter medical information
        
        3. Knowledge Integration
           - Combine healthcare sources
           - Resolve medical data conflicts
           - Track healthcare document lineage
           - Maintain medical data freshness
        
        Output Format (JSON):
        {
            "document_processing": {
                "processed_docs": 150,
                "indexed_topics": ["healthcare_demand", "medical_capacity", "sdoh_impact"],
                "update_timestamp": "2024-03-15T10:30:00Z"
            },
            "retrieval_metrics": {
                "query_relevance": 0.88,
                "context_coverage": 0.92,
                "source_diversity": 0.85
            }
        }
        
        When complete, respond with TERMINATE."""
    )
]

# Prediction Group
prediction_group_agents = [
    AssistantAgent(
        "risk_agent",
        model_client=gemini_client,  # Using Gemini for risk assessment
        description="Handles risk assessment and predictions using Gemini AI",
        system_message="""Generate predictions and risk assessments using Gemini AI's advanced capabilities.
        Your tasks:
        1. Process received data using Gemini's analysis tools
        2. Generate detailed forecasts with confidence intervals
        3. Assess risks using Gemini's risk analysis features
        4. Create comprehensive prediction reports
        5. Send results using DataMessage format
        
        Output Format:
        {
            "predictions": {
                "time_series": [...],
                "confidence_intervals": [...],
                "risk_factors": [...],
                "trend_analysis": {...}
            },
            "risk_assessment": {
                "risk_scores": [...],
                "mitigation_strategies": [...],
                "uncertainty_analysis": {...}
            },
            "recommendations": [...]
        }
        
        When complete, respond with TERMINATE."""
    ),
    AssistantAgent(
        "allocation_agent",
        model_client=model_client,
        description="Plans resource allocation",
        system_message="""Create allocation plans and send upward.
        Your tasks:
        1. Process forecasts
        2. Generate allocation plans
        3. Send plans using DataMessage
        4. Optimize based on feedback"""
    )
]

# Output Group
output_group_agents = [
    AssistantAgent(
        "visualization_agent",
        model_client=model_client,
        description="Creates healthcare data visualizations and sends results upward",
        system_message="""You are the Visualization Agent responsible for creating healthcare data visualizations.
        Your core responsibilities:
        1. Visualization Creation
           - Generate healthcare trend plots
           - Create medical capacity charts
           - Visualize SDoH impacts
           - Display healthcare forecasts
        
        2. Interactive Features
           - Add healthcare data drill-downs
           - Enable medical metric filtering
           - Support healthcare comparisons
           - Include medical data tooltips
        
        3. Layout Design
           - Arrange healthcare visualizations
           - Ensure medical data clarity
           - Maintain consistent healthcare styling
           - Support responsive medical displays
        
        Output Format (JSON):
        {
            "visualizations": [
                {
                    "type": "healthcare_demand",
                    "data": {
                        "x": ["2024-Q1", "2024-Q2", "2024-Q3"],
                        "y": [1200, 1350, 1500],
                        "confidence": {
                            "lower": [1100, 1250, 1400],
                            "upper": [1300, 1450, 1600]
                        }
                    },
                    "layout": {
                        "title": "Healthcare Demand Forecast by Quarter",
                        "height": 400,
                        "width": 800
                    }
                }
            ],
            "interactivity": {
                "zoom": true,
                "filter": true,
                "tooltip": true,
                "export": true
            }
        }
        
        When complete, respond with TERMINATE."""
    ),
    AssistantAgent(
        "critic_agent",
        model_client=model_client,
        description="Reviews healthcare outputs and ensures quality standards",
        system_message="""You are the Critic Agent responsible for reviewing healthcare outputs.
        Your core responsibilities:
        1. Output Review
           - Validate healthcare predictions
           - Check medical methodology
           - Assess healthcare completeness
           - Verify medical accuracy
        
        2. Quality Assurance
           - Monitor healthcare standards
           - Track medical consistency
           - Measure healthcare reliability
           - Report medical issues
        
        3. Improvement Suggestions
           - Recommend healthcare enhancements
           - Identify medical gaps
           - Propose healthcare improvements
           - Optimize medical outputs
        
        Output Format (JSON):
        {
            "review_results": {
                "prediction_quality": {
                    "accuracy": 0.88,
                    "completeness": 0.92,
                    "consistency": 0.90
                },
                "methodology_assessment": {
                    "data_coverage": 0.85,
                    "model_validity": 0.90,
                    "assumption_check": 0.88
                }
            },
            "improvement_suggestions": [
                {
                    "component": "healthcare_forecast",
                    "issue": "seasonal_variation",
                    "impact": "high",
                    "recommendation": "Include holiday effects on medical demand"
                }
            ]
        }
        
        When complete, respond with TERMINATE."""
    )
]

# Create hierarchical group chats
data_group = GroupChat(
    agents=data_group_agents,
    messages=[],
    max_round=5
)

prediction_group = GroupChat(
    agents=prediction_group_agents,
    messages=[],
    max_round=5
)

output_group = GroupChat(
    agents=output_group_agents,
    messages=[],
    max_round=5
)

# Create the manager group that coordinates all other groups
manager_group = GroupChat(
    agents=[manager_agent],
    messages=[],
    max_round=10
)

class PipelineManager:
    def __init__(self):
        self.manager = manager_group
        self.data_group = data_group
        self.prediction_group = prediction_group
        self.output_group = output_group
        self.message_queue = []

    async def process_task(self, task: str):
        # Initialize task with manager
        await self.manager.run_stream(task)
        
        # Process through data group with Gemini-powered research
        data_result = await self.data_group.run_stream(task)
        self.message_queue.append(
            DataMessage("data_group", "processed_data", data_result)
        )
        
        # Process through prediction group with Gemini-powered risk assessment
        pred_result = await self.prediction_group.run_stream(
            json.dumps({"task": task, "data": self.message_queue[-1].to_json()})
        )
        self.message_queue.append(
            DataMessage("prediction_group", "predictions", pred_result)
        )
        
        # Process through output group
        final_result = await self.output_group.run_stream(
            json.dumps({"task": task, "predictions": self.message_queue[-1].to_json()})
        )
        
        # Final manager review
        await self.manager.run_stream(
            json.dumps({"task": task, "final_result": final_result})
        )

async def run_pipeline(task: str):
    pipeline = PipelineManager()
    await pipeline.process_task(task)
    await model_client.close()
    await gemini_client.close()

# Example implementation
async def main():
    print("Starting hierarchical pipeline analysis with Gemini AI integration...")
    task = """
    Analyze resource allocation needs for homeless shelters in Seattle.
    Requirements:
    1. Data Collection (Using Gemini AI):
       - Current shelter occupancy
       - Weather forecasts
       - Eviction trends
       - Economic indicators
       - Social service availability
    2. Predictions (Using Gemini AI):
       - Demand forecast with confidence intervals
       - Risk assessment with mitigation strategies
       - Trend analysis with uncertainty quantification
    3. Output:
       - Resource allocation plan
       - Visualization of trends
       - Quality assurance review
    Budget: $500,000
    Timeline: Next 3 months
    """
    await run_pipeline(task)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 