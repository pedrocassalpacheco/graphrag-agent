# Add these imports at the top of your file
from selenium.webdriver.common.by import By
from graphrag_agent.tools.browser_automation import BrowserAutomation
import time


# Add this method to your FlowGenerator class
async def launch_browser_automation(self, flow_code=None):
    """
    Launch browser automation to visualize the flow in the Langflow UI.

    Args:
        flow_code: Optional flow code to visualize
    """
    logger.info("Launching browser automation...")

    # Create browser automation instance
    browser = BrowserAutomation(headless=False)  # Set to True to run invisibly

    try:
        # Start browser and navigate to localhost
        browser.start("http://localhost:3000/flows")

        # Wait for page to load
        time.sleep(5)

        # Click the "Create first flow" button
        browser.click_element("new-project-btn", By.ID)
        time.sleep(5)

        # Click the "blank flow" button
        browser.click_element("[data-testid='blank-flow']")
        time.sleep(5)

        # If flow_code is provided, you could potentially parse it to add appropriate components
        # For now, just add a Text Input component as an example
        browser.input_text("[data-testid='sidebar-search-input']", "TextInput")
        time.sleep(5)
        browser.click_element("[data-testid='add-component-button-text-input']")

        # Print instructions
        print("\nBrowser is open with Langflow UI.")
        print("The automation has added a TextInput component.")
        print("Press Enter to close the browser and return to the conversation.")

        # Wait for user to press Enter
        input()

    except Exception as e:
        logger.error(f"Error in browser automation: {e}")
        print(f"Error launching browser: {e}")
    finally:
        # Close the browser when done
        browser.close()
