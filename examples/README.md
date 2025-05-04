# Web Crawler Examples

This directory contains two different implementations of web crawlers:

## 1. BeautifulSoup Implementation (`crawler_example.py`)

A simple, straightforward implementation using BeautifulSoup and requests. This is ideal for:
- Small to medium-sized websites
- Quick prototyping
- Simple crawling needs
- Learning purposes

Features:
- Easy to understand and modify
- Minimal dependencies
- Good for basic content extraction
- Suitable for single-domain crawling

## 2. Scrapy Implementation (`web_crawler_demo.ipynb`)

A more robust implementation using the Scrapy framework. This is ideal for:
- Large-scale crawling
- Complex crawling scenarios
- Production environments
- High-performance requirements

Features:
- Built-in support for:
  - Concurrent requests
  - Automatic retries
  - Request throttling
  - Pipeline processing
  - Middleware support
- Better performance
- More scalable
- Built-in support for various output formats

## Usage

### BeautifulSoup Implementation
```bash
python crawler_example.py
```

### Scrapy Implementation
```bash
jupyter notebook web_crawler_demo.ipynb
```

## Requirements

Both implementations require the packages listed in the root `requirements.txt`:
- requests
- beautifulsoup4
- urllib3
- jupyter
- notebook
- scrapy

Install them using:
```bash
pip install -r requirements.txt
``` 