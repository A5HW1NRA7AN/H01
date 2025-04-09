from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uvicorn
from news_newsletter import NewsNewsletter
import time
import json
from datetime import datetime

app = FastAPI(title="AI News Newsletter API")

# Initialize the newsletter generator
newsletter_generator = NewsNewsletter()

# Store for generated newsletters
newsletter_store = []

class NewsletterRequest(BaseModel):
    topic: str
    interval_minutes: Optional[int] = 60  # Default to 1 hour interval

class NewsletterResponse(BaseModel):
    id: str
    topic: str
    content: str
    timestamp: str
    status: str

@app.post("/start-newsletter-generation", response_model=dict)
async def start_newsletter_generation(request: NewsletterRequest, background_tasks: BackgroundTasks):
    """
    Start continuous newsletter generation for a specific topic at regular intervals.
    """
    # Generate a unique ID for this newsletter generation task
    task_id = f"task_{int(time.time())}"
    
    # Add the task to background tasks
    background_tasks.add_task(
        continuous_newsletter_generation, 
        task_id, 
        request.topic, 
        request.interval_minutes
    )
    
    return {
        "task_id": task_id,
        "message": f"Started continuous newsletter generation for topic: {request.topic}",
        "interval_minutes": request.interval_minutes
    }

async def continuous_newsletter_generation(task_id: str, topic: str, interval_minutes: int):
    """
    Continuously generate newsletters at the specified interval.
    """
    while True:
        try:
            # Generate the newsletter
            newsletter = await newsletter_generator.generate_newsletter(topic)
            
            # Create a response object
            response = NewsletterResponse(
                id=task_id,
                topic=topic,
                content=newsletter,
                timestamp=datetime.now().isoformat(),
                status="completed"
            )
            
            # Store the newsletter
            newsletter_store.append(response.dict())
            
            # Wait for the specified interval
            await asyncio.sleep(interval_minutes * 60)
            
        except Exception as e:
            # Log the error and continue
            print(f"Error generating newsletter: {str(e)}")
            await asyncio.sleep(60)  # Wait a minute before retrying

@app.get("/newsletters", response_model=List[NewsletterResponse])
async def get_newsletters():
    """
    Get all generated newsletters.
    """
    return newsletter_store

@app.get("/newsletters/{task_id}", response_model=List[NewsletterResponse])
async def get_newsletters_by_task(task_id: str):
    """
    Get newsletters generated for a specific task.
    """
    return [n for n in newsletter_store if n["id"] == task_id]

@app.delete("/newsletters/{task_id}")
async def stop_newsletter_generation(task_id: str):
    """
    Stop newsletter generation for a specific task.
    Note: This is a placeholder as actual task cancellation would require more complex implementation.
    """
    # In a real implementation, you would need to track and cancel the background task
    return {"message": f"Stopped newsletter generation for task: {task_id}"}

if __name__ == "__main__":
    uvicorn.run("news_newsletter_api:app", host="0.0.0.0", port=8000, reload=True) 