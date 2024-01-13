from pywinauto import Desktop, Application
import ctypes
import pyautogui
import time

def click_at(x, y, clicks=1, button='left'):
    """
    Click at the specified coordinates.
    :param x: X coordinate
    :param y: Y coordinate
    :param clicks: Number of clicks (2 for double-click)
    :param button: 'left' for left click, 'right' for right click
    """
    pyautogui.click(x, y, clicks=clicks, button=button)

def type_text(text, interval=0.05):
    """
    Type a string as if it's being typed on a keyboard.
    :param text: Text to type
    :param interval: Time interval between key presses
    """
    pyautogui.write(text, interval=interval)


def list_controls(window, indent=0):
    """
    Recursively lists controls and their information in a given window.
    
    :param window: The window whose controls are to be listed.
    :param indent: Indentation level (used for recursive calls to format output).
    :return: String containing information about all controls in the window.
    """
    children = window.children()
    controls_info = ''
    for child in children:
        control_type = child.element_info.control_type
        control_info = f"{' ' * indent}{child.window_text()} {control_type} {child.rectangle()}\n"
        controls_info += control_info
        controls_info += list_controls(child, indent + 4)
    return controls_info

def list_windows():
    """
    Lists all visible windows and their basic information.
    
    :return: List of window objects representing all visible windows.
    """
    windows = Desktop(backend="uia").windows()
    windows_list = []
    for i, w in enumerate(windows):
        if w.is_visible() and w.window_text():
            window_info = f"{i}: {w.window_text()}" + (" (current)" if w.handle == ctypes.windll.user32.GetForegroundWindow() else "")
            print(window_info)
            windows_list.append(w)
    return windows_list

def get_window_controls(index, windows):
    """
    Retrieves and prints UI elements of a specified window based on the given index.
    
    :param index: Index of the window in the provided windows list.
    :param windows: List of window objects.
    :return: String containing information about UI elements of the selected window.
    """
    if 0 <= index < len(windows):
        selected_window = windows[index]
        app = Application(backend="uia").connect(handle=selected_window.handle)
        window = app.window(handle=selected_window.handle)
        controls = list_controls(window)
        print(f"\nUI elements for: {selected_window.window_text()}\n")
        print(controls)
        return controls
    else:
        print("Invalid choice. Exiting.")
        return ""



from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time



def open_website_or_query(driver, query):
    """
    Opens a website in the browser controlled by the driver, or performs a Google search if a non-URL query is provided.
    
    :param driver: Selenium WebDriver instance.
    :param query: URL or search query.
    """
    if query.startswith("http://") or query.startswith("https://"):
        driver.get(query)
    else:
        driver.get("https://www.google.com")
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query + Keys.RETURN)

def list_ui_elements(driver, truncate_length=30):
    """
    Lists all UI elements in the current page of the browser and returns their information as a string.
    
    :param driver: Selenium WebDriver instance.
    :param truncate_length: Length at which to truncate the text content of elements.
    :return: String containing information about all UI elements on the page.
    """
    elements = driver.find_elements(By.XPATH, "//*")
    elements_info = ""
    for i, element in enumerate(elements):
        text_content = element.text
        truncated_text = (text_content[:truncate_length] + '...') if len(text_content) > truncate_length else text_content
        element_info = f"{i}: {element.tag_name}, class: {element.get_attribute('class')}, id: {element.get_attribute('id')}, text: {truncated_text}\n"
        elements_info += element_info
    return elements_info

def click_on_ui_element(driver, index):
    """
    Clicks on a UI element in the current page of the browser based on its index.
    
    :param driver: Selenium WebDriver instance.
    :param index: Index of the element to click.
    """
    elements = driver.find_elements(By.XPATH, "//*")
    if 0 <= index < len(elements):
        elements[index].click()

def type_on_ui_element(driver, index, text):
    """
    Types text into a UI element in the current page of the browser based on its index.
    
    :param driver: Selenium WebDriver instance.
    :param index: Index of the element where text is to be typed.
    :param text: Text to type into the element.
    """
    elements = driver.find_elements(By.XPATH, "//*")
    if 0 <= index < len(elements):
        elements[index].send_keys(text)

# Example Usage
driver = webdriver.Chrome()  # Or use another driver like Firefox



###cmd
import subprocess

def run_cmd_command(command):
    """
    Executes a given command in the Windows Command Prompt (CMD).

    :param command: The command to be executed.
    :return: The output and error (if any) as a string.
    """
    try:
        result = subprocess.run(command, check=True, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

def run_powershell_command(command):
    """
    Executes a given command in Windows PowerShell.

    :param command: The PowerShell command or script to be executed.
    :return: The output and error (if any) as a string.
    """
    try:
        ps_command = f"powershell -Command {command}"
        result = subprocess.run(ps_command, check=True, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
