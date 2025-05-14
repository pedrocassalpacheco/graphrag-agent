from graphrag_agent.tools.browser_automation import BrowserAutomation
from selenium.webdriver.common.by import By
import time
import os

# Create browser automation instance
browser = BrowserAutomation(headless=False)  # Set to True to run invisibly

# Start browser and navigate to localhost
browser.start("http://localhost:3000/flows")

try:
    # Wait for page to load
    time.sleep(5)
    # Click the "Create first flow" button
    browser.click_element("new-project-btn", By.ID)
    time.sleep(5)
    # Click the "Create first flow" button
    browser.click_element("[data-testid='blank-flow']")
    time.sleep(5)
    browser.input_text("[data-testid='sidebar-search-input']", "TextInput")
    time.sleep(5)
    browser.click_element("[data-testid='add-component-button-text-input']")
    # Keep the script running until manually terminated
    print("Browser is open. Press Ctrl+C to exit...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nScript terminated by user")
    # Optionally close the browser when the user stops the script
    browser.close()
finally:
    # Don't close the browser
    pass
