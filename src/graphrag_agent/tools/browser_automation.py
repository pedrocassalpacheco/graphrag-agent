import time
from typing import Optional, List, Dict, Any, Union
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class BrowserAutomation:
    """
    Tool for browser automation using Selenium.

    This class provides functionality to interact with web pages through
    a browser session, with capabilities for navigation, element interaction,
    and data extraction.

    Attributes:
        browser (str): The browser to use ('chrome' or 'firefox')
        headless (bool): Whether to run the browser in headless mode
        driver: The Selenium WebDriver instance
    """

    def __init__(
        self, browser: str = "chrome", headless: bool = False, default_timeout: int = 10
    ):
        """
        Initialize the browser automation tool.

        Args:
            browser: Browser to use ('chrome' or 'firefox')
            headless: Whether to run in headless mode
            default_timeout: Default timeout for wait operations in seconds
        """
        self.browser = browser.lower()
        self.headless = headless
        self.default_timeout = default_timeout
        self.driver = None

    def start(self, url: Optional[str] = None) -> bool:
        """
        Start the browser session.

        Args:
            url: Optional initial URL to navigate to

        Returns:
            bool: True if browser started successfully, False otherwise
        """
        try:
            if self.browser == "chrome":
                # Configure Chrome options
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")

                # Set up Chrome driver
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)

            elif self.browser == "firefox":
                # Configure Firefox options
                options = webdriver.FirefoxOptions()
                if self.headless:
                    options.add_argument("--headless")

                # Set up Firefox driver
                service = Service(GeckoDriverManager().install())
                self.driver = webdriver.Firefox(service=service, options=options)

            else:
                logger.error(f"Unsupported browser: {self.browser}")
                return False

            # Navigate to URL if provided
            if url:
                self.navigate(url)

            logger.info(f"Browser session started: {self.browser}")
            return True

        except WebDriverException as e:
            logger.error(f"Failed to start browser: {str(e)}")
            return False

    def navigate(self, url: str) -> bool:
        """
        Navigate to the specified URL.

        Args:
            url: The URL to navigate to

        Returns:
            bool: True if navigation successful, False otherwise
        """
        if not self.driver:
            logger.error("Browser not started. Call start() first.")
            return False

        try:
            logger.info(f"Navigating to: {url}")
            self.driver.get(url)
            return True
        except WebDriverException as e:
            logger.error(f"Navigation failed: {str(e)}")
            return False

    def wait_for_element(
        self, selector: str, by: By = By.CSS_SELECTOR, timeout: Optional[int] = None
    ) -> Optional[webdriver.remote.webelement.WebElement]:
        """
        Wait for an element to be present in the DOM.

        Args:
            selector: Element selector
            by: Selector type (By.CSS_SELECTOR, By.ID, etc.)
            timeout: Wait timeout in seconds

        Returns:
            WebElement if found, None otherwise
        """
        if not self.driver:
            logger.error("Browser not started. Call start() first.")
            return None

        timeout = timeout or self.default_timeout

        try:
            logger.debug(f"Waiting for element: {selector}")
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            logger.warning(f"Timeout waiting for element: {selector}")
            return None

    def click_element(
        self, selector: str, by: By = By.CSS_SELECTOR, timeout: Optional[int] = None
    ) -> bool:
        """
        Click on an element.

        Args:
            selector: Element selector
            by: Selector type
            timeout: Wait timeout in seconds

        Returns:
            bool: True if click successful, False otherwise
        """
        element = self.wait_for_element(selector, by, timeout)

        if not element:
            return False

        try:
            logger.debug(f"Clicking element: {selector}")
            element.click()
            return True
        except (ElementNotInteractableException, WebDriverException) as e:
            logger.error(f"Failed to click element {selector}: {str(e)}")
            return False

    def input_text(
        self,
        selector: str,
        text: str,
        by: By = By.CSS_SELECTOR,
        clear_first: bool = True,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Input text into an element.

        Args:
            selector: Element selector
            text: Text to input
            by: Selector type
            clear_first: Whether to clear the field first
            timeout: Wait timeout in seconds

        Returns:
            bool: True if input successful, False otherwise
        """
        element = self.wait_for_element(selector, by, timeout)

        if not element:
            return False

        try:
            if clear_first:
                element.clear()

            element.send_keys(text)
            logger.debug(f"Input text into {selector}: {text}")
            return True
        except (ElementNotInteractableException, WebDriverException) as e:
            logger.error(f"Failed to input text into {selector}: {str(e)}")
            return False

    def get_text(
        self, selector: str, by: By = By.CSS_SELECTOR, timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Get text from an element.

        Args:
            selector: Element selector
            by: Selector type
            timeout: Wait timeout in seconds

        Returns:
            str: Element text or None if not found
        """
        element = self.wait_for_element(selector, by, timeout)

        if not element:
            return None

        try:
            text = element.text
            logger.debug(f"Got text from {selector}: {text}")
            return text
        except WebDriverException as e:
            logger.error(f"Failed to get text from {selector}: {str(e)}")
            return None

    def get_attribute(
        self,
        selector: str,
        attribute: str,
        by: By = By.CSS_SELECTOR,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """
        Get attribute value from an element.

        Args:
            selector: Element selector
            attribute: Attribute name
            by: Selector type
            timeout: Wait timeout in seconds

        Returns:
            str: Attribute value or None if not found
        """
        element = self.wait_for_element(selector, by, timeout)

        if not element:
            return None

        try:
            value = element.get_attribute(attribute)
            logger.debug(f"Got attribute {attribute} from {selector}: {value}")
            return value
        except WebDriverException as e:
            logger.error(f"Failed to get attribute from {selector}: {str(e)}")
            return None

    def execute_script(self, script: str, *args) -> Any:
        """
        Execute JavaScript in the browser.

        Args:
            script: JavaScript code to execute
            *args: Arguments to pass to the script

        Returns:
            Any: Result of script execution
        """
        if not self.driver:
            logger.error("Browser not started. Call start() first.")
            return None

        try:
            logger.debug(f"Executing script: {script[:50]}...")
            return self.driver.execute_script(script, *args)
        except WebDriverException as e:
            logger.error(f"Failed to execute script: {str(e)}")
            return None

    def take_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Take a screenshot of the current page.

        Args:
            filename: Optional filename for the screenshot

        Returns:
            str: Path to the screenshot file or None if failed
        """
        if not self.driver:
            logger.error("Browser not started. Call start() first.")
            return None

        try:
            if not filename:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"

            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved to: {filename}")
            return filename
        except WebDriverException as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return None

    def get_all_elements(
        self, selector: str, by: By = By.CSS_SELECTOR, timeout: Optional[int] = None
    ) -> List[webdriver.remote.webelement.WebElement]:
        """
        Find all elements matching a selector.

        Args:
            selector: Element selector
            by: Selector type
            timeout: Wait timeout in seconds

        Returns:
            List of WebElements
        """
        if not self.driver:
            logger.error("Browser not started. Call start() first.")
            return []

        timeout = timeout or self.default_timeout

        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            elements = self.driver.find_elements(by, selector)
            logger.debug(f"Found {len(elements)} elements matching: {selector}")
            return elements
        except TimeoutException:
            logger.warning(f"Timeout waiting for elements: {selector}")
            return []
        except WebDriverException as e:
            logger.error(f"Failed to find elements {selector}: {str(e)}")
            return []

    def close(self):
        """Close the browser session."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser session closed")
            except WebDriverException as e:
                logger.error(f"Error closing browser: {str(e)}")
            finally:
                self.driver = None


# Usage example
if __name__ == "__main__":
    # Create browser automation instance
    browser = BrowserAutomation(headless=False)

    try:
        # Start browser and navigate to localhost:3000
        if browser.start("http://localhost:3000/"):
            # Wait for page to load
            time.sleep(2)

            # Example interactions
            # Click a button
            browser.click_element("button.start", By.CSS_SELECTOR)

            # Input text
            browser.input_text("input[name='query']", "Sample query")

            # Get text from an element
            result = browser.get_text(".result-container")
            print(f"Result: {result}")

            # Take a screenshot
            browser.take_screenshot("localhost_capture.png")

            # Wait for user to see the browser
            time.sleep(5)
    finally:
        # Always close the browser
        browser.close()
