import streamlit as st
import requests
import json
import time
from datetime import datetime
import asyncio
import threading

# Set page config
st.set_page_config(
    page_title="AI News Tiles",
    page_icon="📰",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS for tiles
st.markdown("""
<style>
.tile {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease-in-out;
    height: 500px;
    overflow-y: auto;
}
.tile:hover {
    transform: translateY(-5px);
}
.newsletter-section {
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.tile-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #1f77b4;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 0.5rem;
}
.tile-content {
    font-size: 1rem;
    line-height: 1.5;
}
.tile-timestamp {
    font-size: 0.8rem;
    color: #666;
    margin-top: 1rem;
    text-align: right;
}
.tip-box {
    background-color: #e6f7ff;
    border-left: 4px solid #1890ff;
    padding: 10px;
    margin: 10px 0;
    border-radius: 0 4px 4px 0;
}
.point-box {
    background-color: #f6ffed;
    border-left: 4px solid #52c41a;
    padding: 10px;
    margin: 10px 0;
    border-radius: 0 4px 4px 0;
}
.highlight-box {
    background-color: #fff7e6;
    border-left: 4px solid #faad14;
    padding: 10px;
    margin: 10px 0;
    border-radius: 0 4px 4px 0;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📰 AI News Tiles")
st.markdown("""
This app displays three animated tiles with continuously updated news content about Social Determinants of Health (SDoH).
Each tile focuses on a different aspect of the news:
1. **Top Stories**: Latest breaking news and policy changes
2. **Key Highlights**: Important insights, data points, and practical tips
3. **Future Outlook**: Emerging trends, recommendations, and policy interventions
""")

# Input section
with st.form("tiles_form"):
    topic = st.text_input(
        "Enter a topic to research",
        value="Social Determinants of Health (SDoH)",
        placeholder="e.g., Social Determinants of Health (SDoH)"
    )
    submitted = st.form_submit_button("Start Tile Generation")

# Examples
with st.expander("Example Topics"):
    st.markdown("""
    - Social Determinants of Health (SDoH)
    - Healthcare disparities and equity
    - Housing and health outcomes
    - Food insecurity and health
    - Education and health literacy
    - Transportation access and healthcare
    - Environmental health factors
    """)

# Process the request
if submitted and topic:
    try:
        # Start the tile generation for each tile type
        tile_types = ["top_stories", "key_highlights", "future_outlook"]
        task_ids = []
        
        for tile_type in tile_types:
            response = requests.post(
                f"{API_URL}/start-tile-generation",
                json={"topic": topic, "tile_type": tile_type}
            )
            
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                task_ids.append(task_id)
                st.success(f"Started {tile_type} generation for topic: {topic}")
            else:
                st.error(f"Error starting {tile_type} generation: {response.text}")
        
        # Store the task IDs in session state
        if "task_ids" not in st.session_state:
            st.session_state.task_ids = []
        st.session_state.task_ids.extend(task_ids)
        
        # Display a message about the update interval
        st.info("Tiles will be updated every 15 seconds")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Make sure the API server is running at http://localhost:8000")

# Function to fetch and update tile content
def update_tile_content(tile_type):
    try:
        response = requests.get(f"{API_URL}/tiles/{tile_type}/latest")
        if response.status_code == 200:
            content = response.json()
            return content
        else:
            return None
    except Exception as e:
        print(f"Error fetching {tile_type} content: {str(e)}")
        return None

# Function to format content with special styling for tips and points
def format_content(content):
    if not content:
        return "Waiting for content..."
    
    # Replace bullet points with styled boxes
    formatted_content = content
    
    # Format tips
    if "tip:" in formatted_content.lower() or "tips:" in formatted_content.lower():
        lines = formatted_content.split('\n')
        for i, line in enumerate(lines):
            if "tip:" in line.lower() or "tips:" in line.lower():
                lines[i] = f'<div class="tip-box">{line}</div>'
        formatted_content = '\n'.join(lines)
    
    # Format points
    if "point:" in formatted_content.lower() or "points:" in formatted_content.lower():
        lines = formatted_content.split('\n')
        for i, line in enumerate(lines):
            if "point:" in line.lower() or "points:" in line.lower():
                lines[i] = f'<div class="point-box">{line}</div>'
        formatted_content = '\n'.join(lines)
    
    # Format highlights
    if "highlight:" in formatted_content.lower() or "highlights:" in formatted_content.lower():
        lines = formatted_content.split('\n')
        for i, line in enumerate(lines):
            if "highlight:" in line.lower() or "highlights:" in line.lower():
                lines[i] = f'<div class="highlight-box">{line}</div>'
        formatted_content = '\n'.join(lines)
    
    return formatted_content

# Display the tiles
if "task_ids" in st.session_state and st.session_state.task_ids:
    # Create three columns for the tiles
    col1, col2, col3 = st.columns(3)
    
    # Top Stories Tile
    with col1:
        st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
        st.markdown("<div class='tile-header'>Top Stories</div>", unsafe_allow_html=True)
        
        # Fetch and display content
        content = update_tile_content("top_stories")
        if content:
            formatted_content = format_content(content['content'])
            st.markdown(f"<div class='tile-content'>{formatted_content}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='tile-timestamp'>Last updated: {content['timestamp']}</div>", unsafe_allow_html=True)
        else:
            st.info("Waiting for content...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Key Highlights Tile
    with col2:
        st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
        st.markdown("<div class='tile-header'>Key Highlights</div>", unsafe_allow_html=True)
        
        # Fetch and display content
        content = update_tile_content("key_highlights")
        if content:
            formatted_content = format_content(content['content'])
            st.markdown(f"<div class='tile-content'>{formatted_content}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='tile-timestamp'>Last updated: {content['timestamp']}</div>", unsafe_allow_html=True)
        else:
            st.info("Waiting for content...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Future Outlook Tile
    with col3:
        st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
        st.markdown("<div class='tile-header'>Future Outlook</div>", unsafe_allow_html=True)
        
        # Fetch and display content
        content = update_tile_content("future_outlook")
        if content:
            formatted_content = format_content(content['content'])
            st.markdown(f"<div class='tile-content'>{formatted_content}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='tile-timestamp'>Last updated: {content['timestamp']}</div>", unsafe_allow_html=True)
        else:
            st.info("Waiting for content...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add a button to stop all tasks
    if st.button("Stop All Tile Generation"):
        for task_id in st.session_state.task_ids:
            try:
                response = requests.delete(f"{API_URL}/stop-tile-generation/{task_id}")
                if response.status_code == 200:
                    st.success(f"Stopped tile generation for task: {task_id}")
                else:
                    st.error(f"Error stopping task: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Clear the task IDs
        st.session_state.task_ids = []
        st.experimental_rerun()
elif submitted:
    st.warning("Please enter a topic.")

# Auto-refresh the page every 15 seconds
if "task_ids" in st.session_state and st.session_state.task_ids:
    st.markdown("""
    <script>
        setTimeout(function() {
            window.location.reload();
        }, 15000);
    </script>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and AI News Tiles API") 