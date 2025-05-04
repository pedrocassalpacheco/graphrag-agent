from src.utils.web_crawler import WebCrawler

def main():
    # Example usage
    base_url = "https://example.com"  # Replace with your target website
    
    # Create crawler instance
    crawler = WebCrawler(
        base_url=base_url,
        follow_external=False,  # Set to True if you want to follow external links
        max_pages=50,          # Maximum number of pages to crawl
        delay=1.0              # Delay between requests in seconds
    )
    
    # Start crawling
    results = crawler.crawl()
    
    # Save results to a JSON file
    crawler.save_to_file("crawl_results.json")
    
    # Print summary
    print(f"Crawled {len(results)} pages")
    print(f"Results saved to crawl_results.json")

if __name__ == "__main__":
    main() 