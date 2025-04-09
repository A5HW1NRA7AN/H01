from typing import List, Dict
from duckduckgo_search import DDGS
import asyncio
from datetime import datetime
from ollama import OllamaChatCompletionClient

class NewsNewsletter:
    def __init__(self):
        """Initialize the NewsNewsletter."""
        self.model = OllamaChatCompletionClient(model="mistral")
        self.search_client = DDGS()

    async def search_news(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search for news articles using DuckDuckGo."""
        try:
            results = []
            for r in self.search_client.text(query, max_results=max_results):
                results.append({
                    "title": r["title"],
                    "link": r["link"],
                    "snippet": r["body"]
                })
            return results
        except Exception as e:
            print(f"Error searching news: {str(e)}")
            return []

    async def analyze_news(self, news_data: List[Dict], topic: str) -> str:
        """Analyze news articles and provide insights."""
        if not news_data:
            return """
## Top Stories
No news articles found for this topic.

## Key Highlights
No highlights available at this time.

## What's Next
No future outlook available at this time.
"""
        
        # Format the news data for the model
        news_text = "\n\n".join([
            f"Title: {article['title']}\nLink: {article['link']}\nSummary: {article['snippet']}"
            for article in news_data
        ])
        
        # Generate a more comprehensive analysis with tips and points
        prompt = f"""
You are a healthcare policy expert specializing in Social Determinants of Health (SDoH).
Analyze the following news articles about {topic} and provide a comprehensive analysis.

News Articles:
{news_text}

Please provide a detailed analysis with the following sections:

## Top Stories
- List 5-7 key news stories with brief summaries
- For each story, include 2-3 bullet points with key facts
- Highlight any policy changes or initiatives mentioned

## Key Highlights
- Provide 8-10 key insights from the articles
- Include specific data points and statistics when available
- Organize insights into categories (e.g., Economic Factors, Housing, Education, etc.)
- For each insight, provide 1-2 practical tips for healthcare providers or policymakers

## What's Next
- Identify 5-7 emerging trends or developments
- For each trend, provide 2-3 actionable recommendations
- Include a timeline for expected developments
- Suggest potential policy interventions or community programs

Make the content informative, data-driven, and actionable. Focus on practical tips and points that can be implemented by healthcare providers, policymakers, or community organizations.
"""
        
        try:
            response = await self.model.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing news: {str(e)}")
            return """
## Top Stories
Error analyzing news articles.

## Key Highlights
No highlights available at this time.

## What's Next
No future outlook available at this time.
"""

    async def generate_newsletter(self, topic: str) -> str:
        """Generate a newsletter about a given topic."""
        try:
            # Search for news articles
            news_data = await self.search_news(topic)
            
            # Analyze the news data
            insights = await self.analyze_news(news_data, topic)
            
            # Generate the newsletter
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            
            newsletter = f"""
# {topic.title()} Newsletter
{current_date}

{insights}

## Sources
The following sources were used to generate this newsletter:
"""
            
            # Add sources
            for article in news_data:
                newsletter += f"- [{article['title']}]({article['link']})\n"
            
            return newsletter
        except Exception as e:
            print(f"Error generating newsletter: {str(e)}")
            return f"""
# {topic.title()} Newsletter
{datetime.datetime.now().strftime("%B %d, %Y")}

## Top Stories
Error generating newsletter content.

## Key Highlights
No highlights available at this time.

## What's Next
No future outlook available at this time.

## Sources
No sources available at this time.
"""

async def main():
    # Example usage
    newsletter_generator = NewsNewsletter()
    topic = "artificial intelligence latest developments"
    newsletter = await newsletter_generator.generate_newsletter(topic)
    print(newsletter)

if __name__ == "__main__":
    asyncio.run(main()) 