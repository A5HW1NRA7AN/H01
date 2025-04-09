import streamlit as st
import asyncio
import requests
import json
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI News Newsletter",
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
.newsletter-container {
    max-height: 500px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📰 AI News Newsletter")
st.markdown("""
This app uses AI to continuously search for and analyze news articles, then generate newsletters.
Simply enter a topic and interval, and the AI will:
1. Search for 20 relevant news articles at regular intervals
2. Extract key highlights and insights
3. Generate newsletters with top stories and future outlook
""")

# Input section
with st.form("newsletter_form"):
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input(
            "Enter a topic to research",
            placeholder="e.g., artificial intelligence latest developments"
        )
    with col2:
        interval = st.number_input(
            "Generation interval (minutes)",
            min_value=15,
            max_value=1440,
            value=60,
            step=15
        )
    submitted = st.form_submit_button("Start Continuous Newsletter Generation")

# Examples
with st.expander("Example Topics"):
    st.markdown("""
    - artificial intelligence latest developments
    - climate change recent news
    - space exploration breakthroughs
    - healthcare technology innovations
    - quantum computing advances
    - renewable energy trends
    """)

# Process the request
if submitted and topic:
    try:
        # Start the newsletter generation
        response = requests.post(
            f"{API_URL}/start-newsletter-generation",
            json={"topic": topic, "interval_minutes": interval}
        )
        
        if response.status_code == 200:
            task_id = response.json()["task_id"]
            st.success(f"Started continuous newsletter generation for topic: {topic}")
            
            # Store the task ID in session state
            if "task_ids" not in st.session_state:
                st.session_state.task_ids = []
            st.session_state.task_ids.append(task_id)
            
            # Display a message about the interval
            st.info(f"Newsletters will be generated every {interval} minutes")
        else:
            st.error(f"Error starting newsletter generation: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Make sure the API server is running at http://localhost:8000")

# Display active tasks
if "task_ids" in st.session_state and st.session_state.task_ids:
    st.subheader("Active Newsletter Generation Tasks")
    for task_id in st.session_state.task_ids:
        with st.expander(f"Task: {task_id}"):
            try:
                # Get newsletters for this task
                response = requests.get(f"{API_URL}/newsletters/{task_id}")
                if response.status_code == 200:
                    newsletters = response.json()
                    
                    if newsletters:
                        st.markdown(f"**Generated {len(newsletters)} newsletters so far**")
                        
                        # Display the latest newsletter
                        latest = newsletters[-1]
                        st.markdown(f"**Latest newsletter (generated at {latest['timestamp']})**")
                        
                        # Parse the newsletter content
                        content = latest["content"]
                        sections = content.split('\n\n')
                        
                        # Display title and date
                        st.markdown(f"<div class='tile'>{sections[0]}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='tile'>{sections[1]}</div>", unsafe_allow_html=True)
                        
                        # Create columns for the main sections
                        col1, col2, col3 = st.columns(3)
                        
                        # Display sections in tiles with animation
                        with col1:
                            st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
                            st.markdown("## Top Stories")
                            for section in sections:
                                if "## Top Stories" in section:
                                    st.markdown(section.split("## Key Highlights")[0])
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
                            st.markdown("## Key Highlights")
                            for section in sections:
                                if "## Key Highlights" in section:
                                    st.markdown(section.split("## What's Next")[0])
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
                            st.markdown("## What's Next")
                            for section in sections:
                                if "## What's Next" in section:
                                    st.markdown(section.split("## Sources")[0])
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display sources in a separate tile
                        st.markdown("<div class='tile newsletter-section'>", unsafe_allow_html=True)
                        st.markdown("## Sources")
                        for section in sections:
                            if "## Sources" in section:
                                st.markdown(section)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add a download button for the latest newsletter
                        st.download_button(
                            label="📥 Download Latest Newsletter",
                            data=content,
                            file_name=f"ai_news_newsletter_{topic.replace(' ', '_')}_{latest['timestamp']}.md",
                            mime="text/markdown"
                        )
                        
                        # Add a button to stop the task
                        if st.button(f"Stop Generation for Task: {task_id}"):
                            stop_response = requests.delete(f"{API_URL}/newsletters/{task_id}")
                            if stop_response.status_code == 200:
                                st.session_state.task_ids.remove(task_id)
                                st.success(f"Stopped newsletter generation for task: {task_id}")
                            else:
                                st.error(f"Error stopping task: {stop_response.text}")
                    else:
                        st.info("No newsletters generated yet. Please wait...")
                else:
                    st.error(f"Error fetching newsletters: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
elif submitted:
    st.warning("Please enter a topic.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and AI News Newsletter API") 