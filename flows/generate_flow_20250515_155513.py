from langflow.components.youtube import YouTubeCommentsComponent
from langflow.components.inputs import URLInput
from langflow.graph import Graph

def youtube_comments_flow():
    # URL Input Component
    url_input = URLInput()
    
    # YouTube Comments Component
    youtube_comments = YouTubeCommentsComponent()
    youtube_comments.video_url = url_input.url_output  # Connect URL input output to YouTube comments component
    youtube_comments.api_key = "YOUR_YOUTUBE_API_KEY"
    youtube_comments.max_results = 20
    youtube_comments.sort_by = "relevance"
    youtube_comments.include_replies = False
    youtube_comments.include_metrics = True

    # Define the graph
    return Graph(start=url_input, end=youtube_comments)