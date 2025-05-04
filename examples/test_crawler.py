from src.utils.web_crawler import WebCrawler
import logging

def main():
    # Set up logging to see output in real-time
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create crawler instance
    crawler = WebCrawler(
        base_url="https://docs.python.org/3/",  # Using Python docs as an example
        follow_external=False,
        max_pages=10,  # Limit to 10 pages for testing
        delay=1.0  # 1 second delay between requests
    )
    
    # Start crawling
    print("Starting crawl...")
    results = crawler.crawl()
    
    # Save results
    crawler.save_to_file("crawl_results.json")
    print(f"\nCrawl completed. Found {len(results)} pages.")
    print(f"Visited URLs: {len(crawler.visited_urls)}")
    print(f"Results saved to crawl_results.json")

if __name__ == "__main__":
    main() 