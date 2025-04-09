from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
from datetime import datetime
import random
from duckduckgo_search import DDGS
import time
import logging
import json
import os
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI News Tiles API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Rate limiting settings
RATE_LIMIT_DELAY = 5  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds to wait after a rate limit error
CACHE_DURATION = 300  # 5 minutes cache duration

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the search client with rate limiting
class RateLimitedDDGS:
    def __init__(self, delay=5):
        self.ddgs = DDGS()
        self.delay = delay
        self.last_request_time = 0
        self.max_retries = 3
        self.retry_delay = 10
    
    def search(self, query, max_results=15):
        # Ensure delay between requests
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.delay:
            time.sleep(self.delay - time_since_last_request)
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                results = list(self.ddgs.text(query, max_results=max_results))
                self.last_request_time = time.time()
                return results
            except Exception as e:
                logger.warning(f"Search attempt {attempt+1} failed: {str(e)}")
                if "202 Ratelimit" in str(e):
                    logger.info(f"Rate limited, waiting {self.retry_delay} seconds before retry")
                    time.sleep(self.retry_delay)
                else:
                    time.sleep(self.delay)
        
        # If all retries failed, return empty list
        logger.error("All search attempts failed")
        return []

search_client = RateLimitedDDGS(delay=5)

# Cache for search results
@lru_cache(maxsize=100)
def cached_search(query: str, max_results: int = 20):
    """Cache search results to reduce API calls"""
    cache_file = os.path.join(CACHE_DIR, f"{hash(query)}_{max_results}.json")
    
    # Check if cache exists and is fresh
    if os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        if cache_age < CACHE_DURATION:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
    
    # Perform search and cache results
    try:
        results = list(search_client.text(query, max_results=max_results))
        with open(cache_file, 'w') as f:
            json.dump(results, f)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

# Store for generated content for each tile
tile_store = {
    "top_stories": [],
    "key_highlights": [],
    "future_outlook": []
}

# Active tasks
active_tasks = {}

class TileRequest(BaseModel):
    topic: str
    tile_type: str  # "top_stories", "key_highlights", or "future_outlook"

class TileResponse(BaseModel):
    id: str
    topic: str
    tile_type: str
    content: str
    timestamp: str
    status: str

def generate_three_word_title(articles, start_idx, count):
    """Generate a dynamic 3-word title from article content"""
    words = []
    for article in articles[start_idx:start_idx + count]:
        title_words = article['title'].split()
        words.extend([w.lower() for w in title_words if len(w) > 3])
    
    from collections import Counter
    word_counts = Counter(words)
    common_words = [word for word, _ in word_counts.most_common(10)]
    
    selected_words = []
    for word in common_words:
        if len(selected_words) < 3 and word not in selected_words:
            selected_words.append(word)
    
    while len(selected_words) < 3:
        selected_words.append("healthcare")
    
    return ' '.join(selected_words[:3]).title()

def generate_tile_content(topic: str, tile_type: str) -> str:
    """Generate content for a specific tile type"""
    try:
        # Search for news articles
        results = cached_search(f"{topic} news", max_results=20)
        
        if not results:
            logger.warning(f"No results found for topic: {topic}")
            return f"No news articles found for {topic}."
        
        # Generate content based on tile type
        if tile_type == "top_stories":
            dynamic_title = generate_three_word_title(results, 0, 5)
            content = f"{dynamic_title}\n\n"
            for i, article in enumerate(results[:5], 1):
                content += f"{i}. {article['title']}\n"
                content += f"   {article['snippet'][:200]}...\n\n"
        
        elif tile_type == "key_highlights":
            dynamic_title = generate_three_word_title(results, 5, 5)
            content = f"{dynamic_title}\n\n"
            for i, article in enumerate(results[5:10], 1):
                content += f"Point {i}: {article['title']}\n"
                content += f"{article['snippet'][:150]}...\n\n"
        
        else:  # future_outlook
            dynamic_title = generate_three_word_title(results, 10, 5)
            content = f"{dynamic_title}\n\n"
            for i, article in enumerate(results[10:15], 1):
                content += f"Trend {i}: {article['title']}\n"
                content += f"{article['snippet'][:150]}...\n\n"
        
        return content
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return f"Unable to generate content at this time. Please try again later."

@app.post("/start-tile-generation", response_model=dict)
async def start_tile_generation(request: TileRequest, background_tasks: BackgroundTasks):
    """Start continuous content generation for a specific tile type"""
    if request.tile_type not in ["top_stories", "key_highlights", "future_outlook"]:
        return {"error": "Invalid tile type"}
    
    task_id = f"task_{request.tile_type}_{int(datetime.now().timestamp())}"
    
    # Generate initial content
    content = generate_tile_content(request.topic, request.tile_type)
    initial_response = TileResponse(
        id=task_id,
        topic=request.topic,
        tile_type=request.tile_type,
        content=content,
        timestamp=datetime.now().isoformat(),
        status="completed"
    )
    tile_store[request.tile_type].append(initial_response.dict())
    
    background_tasks.add_task(
        continuous_tile_generation,
        task_id,
        request.topic,
        request.tile_type
    )
    
    active_tasks[task_id] = {
        "topic": request.topic,
        "tile_type": request.tile_type,
        "active": True
    }
    
    return {
        "task_id": task_id,
        "message": f"Started continuous content generation for {request.tile_type}",
        "interval_seconds": 10
    }

async def continuous_tile_generation(task_id: str, topic: str, tile_type: str):
    """Continuously generate content for a specific tile type"""
    while active_tasks.get(task_id, {}).get("active", False):
        try:
            content = generate_tile_content(topic, tile_type)
            
            response = TileResponse(
                id=task_id,
                topic=topic,
                tile_type=tile_type,
                content=content,
                timestamp=datetime.now().isoformat(),
                status="completed"
            )
            
            tile_store[tile_type].append(response.dict())
            
            if len(tile_store[tile_type]) > 10:
                tile_store[tile_type] = tile_store[tile_type][-10:]
            
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in continuous generation: {str(e)}")
            await asyncio.sleep(10)

@app.get("/tiles/{tile_type}", response_model=List[TileResponse])
async def get_tile_content(tile_type: str):
    """Get all content for a specific tile type"""
    if tile_type not in ["top_stories", "key_highlights", "future_outlook"]:
        return {"error": "Invalid tile type"}
    return tile_store[tile_type]

@app.get("/tiles/{tile_type}/latest", response_model=TileResponse)
async def get_latest_tile_content(tile_type: str):
    """Get the latest content for a specific tile type"""
    if tile_type not in ["top_stories", "key_highlights", "future_outlook"]:
        return {"error": "Invalid tile type"}
    
    if not tile_store[tile_type]:
        # Return a default response instead of an error
        return TileResponse(
            id=f"default_{tile_type}",
            topic="healthcare",
            tile_type=tile_type,
            content=f"## Loading {tile_type.replace('_', ' ').title()}\n\nGenerating content...",
            timestamp=datetime.now().isoformat(),
            status="pending"
        )
    
    return tile_store[tile_type][-1]

@app.get("/active-tasks", response_model=Dict)
async def get_active_tasks():
    """Get all active tasks"""
    return active_tasks

@app.delete("/stop-tile-generation/{task_id}")
async def stop_tile_generation(task_id: str):
    """Stop content generation for a specific task"""
    if task_id in active_tasks:
        active_tasks[task_id]["active"] = False
        return {"message": f"Stopped content generation for task: {task_id}"}
    return {"error": f"Task not found: {task_id}"}

if __name__ == "__main__":
    uvicorn.run("news_tiles_api:app", host="0.0.0.0", port=8000, reload=True) 