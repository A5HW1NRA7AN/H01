from flask import Flask, render_template, jsonify, request
from news_tiles_api import generate_tile_content, TileResponse
from datetime import datetime, timedelta
import threading
import time
import logging
import os
import json
import random
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyDptAFMnzQVs619i6PGOA1bYLhVSrm7rv0")

# Initialize Gemini Model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize Gemini Model for forecasting
forecast_model = genai.GenerativeModel("gemini-pro")

app = Flask(__name__)

# Store for generated content
tile_store = {
    "top_stories": [],
    "key_highlights": [],
    "future_outlook": []
}

# Active tasks
active_tasks = {}

# Lock for thread-safe operations
tile_lock = threading.Lock()

# Chatbot constants
REALTIME_KEYWORDS = ["now", "current", "latest", "today", "real-time", "live"]

# Configuration
TRUSTED_DOMAINS = ['.gov', '.edu', '.org', '.us', 'nih.gov', 'cdc.gov', 'who.int']

KEYWORDS = [
    "social determinants", "health disparities", "housing", "unemployment",
    "mental health", "food insecurity", "covid-19", "telehealth", "public health",
    "health insurance", "access to care", "environmental justice", "education inequality"
]

def is_realtime_query(message: str) -> bool:
    return any(re.search(rf"\b{kw}\b", message, re.IGNORECASE) for kw in REALTIME_KEYWORDS)

def scrape_context_duckduckgo(query: str) -> str:
    try:
        search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query + ' site:cdc.gov OR site:who.int')}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AutogenBot/1.0)"}
        response = requests.get(search_url, headers=headers, timeout=10)
        links = re.findall(r'href="(https://[^"]+)"', response.text)
        top_links = list(dict.fromkeys(links))[:2]

        scraped_text = ""
        for link in top_links:
            try:
                page = requests.get(link, headers=headers, timeout=10)
                texts = re.findall(r"<p>(.*?)</p>", page.text, re.DOTALL)
                cleaned = "\n".join(re.sub(r"<.*?>", "", t).strip() for t in texts[:3])
                scraped_text += f"\n[From {link}]:\n{cleaned}\n"
            except:
                continue

        return scraped_text if scraped_text else "\n[No useful real-time context found.]"
    except Exception as e:
        return f"\n[Web scraping via DuckDuckGo failed: {str(e)}]"

def generate_reply(message: str) -> str:
    if is_realtime_query(message):
        scraped_context = scrape_context_duckduckgo(message)
        prompt = f"{message}\n\nUse this real-time data for context:\n{scraped_context}"
    else:
        prompt = message

    response = gemini_model.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else str(response)

# Load location data
def load_location_data():
    """Load location data from CSV or generate sample data if file doesn't exist"""
    try:
        # Try to load from CSV
        df = pd.read_csv('data/loc.csv')
    except FileNotFoundError:
        logger.warning("loc.csv not found. Generating sample data...")
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        n_points = 1000
        
        # Generate random coordinates around major cities
        cities = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740)
        }
        
        data = []
        for city, (lat, lng) in cities.items():
            n_city_points = n_points // len(cities)
            city_data = {
                'latitude': np.random.normal(lat, 0.1, n_city_points),
                'longitude': np.random.normal(lng, 0.1, n_city_points),
                'intensity': np.random.uniform(0.5, 1.0, n_city_points)
            }
            data.extend([{
                'latitude': lat,
                'longitude': lng,
                'intensity': intensity
            } for lat, lng, intensity in zip(
                city_data['latitude'],
                city_data['longitude'],
                city_data['intensity']
            )])
        
        df = pd.DataFrame(data)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save sample data
        df.to_csv('data/loc.csv', index=False)
        logger.info("Sample data generated and saved to data/loc.csv")
    
    # Ensure required columns exist
    required_columns = ['latitude', 'longitude']
    if not all(col in df.columns for col in required_columns):
        logger.error("Missing required columns in loc.csv")
        return []
    
    # Clean the data
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)]
    
    # Convert to list of dictionaries
    data = df[required_columns].to_dict('records')
    
    # Add intensity if not present
    for point in data:
        if 'intensity' not in point:
            point['intensity'] = 1.0
    
    logger.info(f"Successfully loaded {len(data)} location points")
    return data

def analyze_location_data():
    """Analyze the location data and return statistics"""
    try:
        df = pd.read_csv('data/loc.csv')
        
        # Basic statistics
        stats = {
            "total_points": len(df),
            "latitude_range": {
                "min": float(df['latitude'].min()),
                "max": float(df['latitude'].max()),
                "mean": float(df['latitude'].mean())
            },
            "longitude_range": {
                "min": float(df['longitude'].min()),
                "max": float(df['longitude'].max()),
                "mean": float(df['longitude'].mean())
            }
        }
        
        # Add intensity statistics if available
        if 'intensity' in df.columns:
            stats["intensity"] = {
                "min": float(df['intensity'].min()),
                "max": float(df['intensity'].max()),
                "mean": float(df['intensity'].mean())
            }
        
        # Calculate point density by region
        df['region'] = pd.cut(df['latitude'], bins=5, labels=['North', 'North-Central', 'Central', 'South-Central', 'South'])
        region_counts = df['region'].value_counts().to_dict()
        stats["region_distribution"] = region_counts
        
        logger.info(f"Analyzed {stats['total_points']} location points")
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing location data: {str(e)}")
        return None

# Initialize location data and statistics
location_data = load_location_data()
location_stats = analyze_location_data()
logger.info(f"Initialized with {len(location_data)} location points")

def background_tile_generation(topic: str, tile_type: str):
    """Background task to continuously generate content"""
    task_id = f"{tile_type}_{topic}"
    logger.info(f"Starting background tile generation for {task_id}")
    
    while active_tasks.get(task_id, {}).get("active", True):
        try:
            content = generate_tile_content(topic, tile_type)
            if content:  # Only update if content was generated
                response = TileResponse(
                    id=f"task_{tile_type}_{int(datetime.now().timestamp())}",
                    topic=topic,
                    tile_type=tile_type,
                    content=content,
                    timestamp=datetime.now().isoformat(),
                    status="completed"
                )
                
                # Thread-safe update of tile_store
                with tile_lock:
                    tile_store[tile_type].append(response.model_dump())
                    if len(tile_store[tile_type]) > 10:
                        tile_store[tile_type] = tile_store[tile_type][-10:]
                
                logger.info(f"Generated new content for {tile_type}")
            else:
                logger.warning(f"No content generated for {tile_type}")
                
        except Exception as e:
            logger.error(f"Error generating content for {tile_type}: {str(e)}")
            # Add a default message when generation fails
            with tile_lock:
                if not tile_store[tile_type]:  # Only add default if no content exists
                    default_response = TileResponse(
                        id=f"default_{tile_type}",
                        topic=topic,
                        tile_type=tile_type,
                        content=f"Loading {tile_type.replace('_', ' ').title()}\n\nGenerating content...",
                        timestamp=datetime.now().isoformat(),
                        status="pending"
                    )
                    tile_store[tile_type].append(default_response.model_dump())
        
        time.sleep(30)  # Increased sleep time to reduce API calls

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')

@app.route('/location_map.html')
def location_map():
    """Serve the location map page"""
    return render_template('location_map.html')

@app.route('/api/heatmap/data')
def get_heatmap_data():
    """Get heatmap data"""
    try:
        return jsonify({
            "status": "success",
            "data": location_data,
            "stats": location_stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting heatmap data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/start-tile-generation', methods=['POST'])
def start_tile_generation():
    """Start content generation for a specific tile type"""
    try:
        data = request.get_json()
        topic = data.get('topic', 'healthcare')
        tile_type = data.get('tile_type')
        
        if tile_type not in ["top_stories", "key_highlights", "future_outlook"]:
            return jsonify({"error": "Invalid tile type"}), 400
        
        # Generate initial content
        content = generate_tile_content(topic, tile_type)
        initial_response = TileResponse(
            id=f"task_{tile_type}_{int(datetime.now().timestamp())}",
            topic=topic,
            tile_type=tile_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            status="completed"
        )
        
        # Thread-safe update
        with tile_lock:
            tile_store[tile_type].append(initial_response.model_dump())
        
        # Start background task
        task_id = f"{tile_type}_{topic}"
        active_tasks[task_id] = {"active": True}
        thread = threading.Thread(
            target=background_tile_generation,
            args=(topic, tile_type)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started tile generation for {task_id}")
        
        return jsonify({
            "task_id": task_id,
            "message": f"Started continuous content generation for {tile_type}",
            "interval_seconds": 30
        })
        
    except Exception as e:
        logger.error(f"Error in start_tile_generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tiles/<tile_type>')
def get_tile_content(tile_type):
    """Get all content for a specific tile type"""
    if tile_type not in ["top_stories", "key_highlights", "future_outlook"]:
        return jsonify({"error": "Invalid tile type"}), 400
    
    with tile_lock:
        return jsonify(tile_store[tile_type])

@app.route('/api/tiles/<tile_type>/latest')
def get_latest_tile_content(tile_type):
    """Get the latest content for a specific tile type"""
    if tile_type not in ["top_stories", "key_highlights", "future_outlook"]:
        return jsonify({"error": "Invalid tile type"}), 400
    
    with tile_lock:
        if not tile_store[tile_type]:
            return jsonify(TileResponse(
                id=f"default_{tile_type}",
                topic="healthcare",
                tile_type=tile_type,
                content=f"Loading {tile_type.replace('_', ' ').title()}\n\nGenerating content...",
                timestamp=datetime.now().isoformat(),
                status="pending"
            ).model_dump())
        
        return jsonify(tile_store[tile_type][-1])

@app.route('/api/active-tasks')
def get_active_tasks():
    """Get all active tasks"""
    return jsonify(active_tasks)

@app.route('/api/stop-tile-generation/<task_id>', methods=['DELETE'])
def stop_tile_generation(task_id):
    """Stop content generation for a specific task"""
    if task_id in active_tasks:
        active_tasks[task_id]["active"] = False
        logger.info(f"Stopped tile generation for {task_id}")
        return jsonify({"message": f"Stopped content generation for task: {task_id}"})
    return jsonify({"error": f"Task not found: {task_id}"}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(active_tasks),
        "tile_counts": {k: len(v) for k, v in tile_store.items()}
    })

@app.route('/api/healthcare-facilities/data')
def get_healthcare_facilities_data():
    try:
        app.logger.info("Reading healthcare facilities data from CSV file")
        # Read data from CSV file
        df = pd.read_csv('data/test.csv')
        app.logger.info(f"Successfully loaded CSV with {len(df)} rows")
        
        # Get the national data (first row after header)
        national_data = df.iloc[0]
        app.logger.info("Extracted national data from first row")
        
        # Process the data
        app.logger.info("Processing healthcare facilities data")
        
        # Return data in the format expected by the frontend
        return jsonify({
            'total_facilities': int(national_data['Number of facilities']),
            'states': int(national_data['Number of states and territories with operations']),
            'special_focus_facilities': int(national_data['Number of Special Focus Facilities (SFF)']),
            'sff_candidates': int(national_data['Number of SFF candidates']),
            'abuse_reports': int(national_data['Number of facilities with an abuse icon']),
            'abuse_percentage': float(national_data['Percentage of facilities with an abuse icon']),
            'quality_metrics': {
                'overall_rating': float(national_data['Average overall 5-star rating']),
                'health_inspection': float(national_data['Average health inspection rating']),
                'staffing_rating': float(national_data['Average staffing rating']),
                'quality_rating': float(national_data['Average quality rating'])
            },
            'staffing_metrics': {
                'nurse_hours': float(national_data['Average total nurse hours per resident day']),
                'weekend_hours': float(national_data['Average total weekend nurse hours per resident day']),
                'rn_hours': float(national_data['Average total Registered Nurse hours per resident day']),
                'nurse_turnover': float(national_data['Average total nursing staff turnover percentage']),
                'rn_turnover': float(national_data['Average Registered Nurse turnover percentage']),
                'admin_changes': float(national_data['Average number of administrators who have left the nursing home'])
            },
            'performance_metrics': {
                'readmission_rate': float(national_data['Average percentage of short-stay residents who were re-hospitalized after a nursing home admission']),
                'pressure_ulcers': float(national_data['Average percentage of long-stay residents with pressure ulcers']),
                'falls': float(national_data['Average percentage of long-stay residents experiencing one or more falls with major injury']),
                'uti': float(national_data['Average percentage of long-stay residents with a urinary tract infection']),
                'depression': float(national_data['Average percentage of long-stay residents who have symptoms of depression']),
                'covid_vaccination': float(national_data['Average percentage of current residents up to date with COVID-19 vaccines']),
                'staff_vaccination': float(national_data['Average percentage of healthcare personnel up to date with COVID-19 vaccines'])
            },
            'compliance_metrics': {
                'total_fines': int(national_data['Total number of fines']),
                'avg_fines': float(national_data['Average number of fines']),
                'total_fine_amount': float(national_data['Total amount of fines in dollars']),
                'avg_fine_amount': float(national_data['Average amount of fines in dollars']),
                'payment_denials': int(national_data['Total number of payment denials']),
                'avg_denials': float(national_data['Average number of payment denials'])
            }
        })

    except FileNotFoundError as e:
        app.logger.error(f"CSV file not found: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"CSV file not found: {str(e)}"
        }), 500
    except pd.errors.EmptyDataError as e:
        app.logger.error(f"CSV file is empty: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"CSV file is empty: {str(e)}"
        }), 500
    except KeyError as e:
        app.logger.error(f"Missing column in CSV: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Missing column in CSV: {str(e)}"
        }), 500
    except ValueError as e:
        app.logger.error(f"Error converting value: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error converting value: {str(e)}"
        }), 500
    except Exception as e:
        app.logger.error(f"Error processing healthcare facilities data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route("/api/chat/query", methods=["POST"])
def handle_chat_query():
    """Handle chatbot queries"""
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        result = generate_reply(user_message)
        return jsonify({"response": result})
    except Exception as e:
        logger.error(f"Error handling chat query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat/query", methods=["GET"])
def handle_chat_query_get():
    """Handle chatbot queries via GET (for testing)"""
    try:
        user_message = request.args.get("message", "")
        if not user_message:
            return jsonify({"error": "No message parameter provided"}), 400

        result = generate_reply(user_message)
        return jsonify({"response": result})
    except Exception as e:
        logger.error(f"Error handling chat query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/unemployment/data')
def get_unemployment_data():
    """Get unemployment rate data for the chart"""
    try:
        df = pd.read_csv('data/unemp.csv')
        
        # Get the last 12 months of data
        recent_data = df.tail(12)
        
        # Format the data for the chart
        data = {
            'labels': [f"{row['Year']}-{row['Month']}" for _, row in recent_data.iterrows()],
            'unemployment_rate': recent_data['Unemployment_Rate'].tolist(),
            'industrial_production': recent_data['Industrial_Production'].tolist(),
            'consumer_price_index': recent_data['Consumer_Price Index'].tolist(),
            'retail_sales': recent_data['Retail_Sales'].tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        logger.error(f"Error getting unemployment data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/unemployment/forecast')
def get_unemployment_forecast():
    """Get unemployment rate forecast data with enhanced analytics"""
    try:
        # Read historical data
        df = pd.read_csv('data/unemp.csv')
        recent_data = df.tail(24)  # Last 24 months for better trend analysis
        
        # Prepare data for forecasting
        unemployment_data = recent_data['Unemployment_Rate'].tolist()
        dates = [f"{row['Year']}-{row['Month']}" for _, row in recent_data.iterrows()]
        
        # Calculate additional metrics
        seasonal_pattern = np.array(unemployment_data[-12:]) - np.array(unemployment_data[-24:-12])
        yoy_change = np.mean(seasonal_pattern)
        volatility = np.std(unemployment_data[-12:])
        
        try:
            # Try to get forecast from Gemini
            data_context = "\n".join([f"{date}: {rate}%" for date, rate in zip(dates[-12:], unemployment_data[-12:])])
            forecast_prompt = f"""
            Based on this unemployment rate data for the past 12 months:
            {data_context}
            
            Additional context:
            - Year-over-year change: {yoy_change:.2f}%
            - Recent volatility: {volatility:.2f}
            - Seasonal pattern observed: {', '.join([f'{x:.2f}%' for x in seasonal_pattern])}
            
            Analyze the trends and provide:
            1. Next 3 months forecast with confidence levels
            2. Key factors influencing the forecast
            3. Potential economic impacts
            4. Seasonal adjustments
            5. Risk factors that could affect the forecast
            
            Format the response as JSON with these keys:
            - forecast_values (array of 3 numbers)
            - confidence_levels (array of 3 numbers between 0-100)
            - factors (array of strings)
            - impacts (array of strings)
            - seasonal_adjustments (array of 3 numbers)
            - risk_factors (array of strings)
            """
            
            # Get forecast from Gemini if available
            if 'forecast_model' in globals():
                response = forecast_model.generate_content(forecast_prompt)
                forecast_data = json.loads(response.text)
            else:
                # Fallback to simple forecasting if Gemini is not available
                recent_values = unemployment_data[-6:]
                slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                last_value = recent_values[-1]
                
                forecast_data = {
                    'forecast_values': [
                        last_value + slope,
                        last_value + slope * 2,
                        last_value + slope * 3
                    ],
                    'confidence_levels': [85, 75, 65],
                    'factors': [
                        'Historical trend analysis',
                        'Recent momentum',
                        'Seasonal patterns'
                    ],
                    'impacts': [
                        'Potential economic growth changes',
                        'Labor market adjustments',
                        'Policy implications'
                    ],
                    'seasonal_adjustments': [
                        np.mean(seasonal_pattern[:4]),
                        np.mean(seasonal_pattern[4:8]),
                        np.mean(seasonal_pattern[8:])
                    ],
                    'risk_factors': [
                        'Economic policy changes',
                        'Market volatility',
                        'Global economic conditions'
                    ]
                }
        except Exception as e:
            app.logger.warning(f"Error with forecast generation, using fallback: {str(e)}")
            # Use simple moving average for forecast
            recent_values = unemployment_data[-6:]
            current_trend = np.mean(np.diff(recent_values))
            last_value = recent_values[-1]
            
            forecast_data = {
                'forecast_values': [
                    last_value + current_trend,
                    last_value + current_trend * 2,
                    last_value + current_trend * 3
                ],
                'confidence_levels': [80, 70, 60],
                'factors': ['Historical trend'],
                'impacts': ['Economic outlook'],
                'seasonal_adjustments': [0, 0, 0],
                'risk_factors': ['Forecast uncertainty']
            }
        
        # Calculate trend indicators
        recent_trend = np.polyfit(range(6), unemployment_data[-6:], 1)[0]
        current_momentum = np.mean(np.diff(unemployment_data[-3:]))
        trend_strength = abs(recent_trend) * (1 + abs(current_momentum))
        
        # Calculate seasonally adjusted forecast
        seasonal_adjustments = forecast_data.get('seasonal_adjustments', [0, 0, 0])
        adjusted_forecast = [f + s for f, s in zip(forecast_data['forecast_values'], seasonal_adjustments)]
        
        # Prepare next three months dates
        next_three_months = []
        for i in range(1, 4):
            next_date = pd.Timestamp(dates[-1]) + pd.DateOffset(months=i)
            next_three_months.append(next_date.strftime("%Y-%m"))
        
        return jsonify({
            'status': 'success',
            'data': {
                'historical_dates': dates[-12:],
                'historical_values': unemployment_data[-12:],
                'forecast_dates': next_three_months,
                'forecast_values': forecast_data['forecast_values'],
                'adjusted_forecast': adjusted_forecast,
                'confidence_levels': forecast_data['confidence_levels'],
                'trend_direction': 'up' if recent_trend > 0 else 'down',
                'trend_strength': float(trend_strength),
                'trend_momentum': float(current_momentum),
                'year_over_year_change': float(yoy_change),
                'volatility': float(volatility),
                'influencing_factors': forecast_data['factors'][:3],
                'economic_impacts': forecast_data['impacts'][:3],
                'risk_factors': forecast_data.get('risk_factors', [])[:3],
                'seasonal_adjustments': seasonal_adjustments
            }
        })
    except Exception as e:
        app.logger.error(f"Error generating forecast: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Start initial tile generation
    topics = ['healthcare', 'medical technology', 'healthcare policy']
    tile_types = ['top_stories', 'key_highlights', 'future_outlook']
    
    logger.info("Starting initial tile generation")
    for topic, tile_type in zip(topics, tile_types):
        thread = threading.Thread(
            target=background_tile_generation,
            args=(topic, tile_type)
        )
        thread.daemon = True
        thread.start()
        logger.info(f"Started thread for {tile_type}")
    
    # Run Flask app
    logger.info("Starting Flask app on port 5000")
    app.run(debug=True, port=5000) 