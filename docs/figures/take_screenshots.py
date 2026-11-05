#!/usr/bin/env python3
"""
Screenshots of the running app for the research paper.

Start the app first (./run.sh --fast --no-open), then:

    python docs/figures/take_screenshots.py
"""

from pathlib import Path

from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
URL = "http://localhost:5173"

QUESTIONS = [
    # (file name, language button, question)
    ("ui_hindi_answer.png", "Hindi", "अगली परीक्षा कब है"),
    ("ui_kannada_answer.png", "Kannada", "ಬಿಸಿಎ ಶುಲ್ಕ ಎಷ್ಟು"),
    ("ui_tamil_answer.png", "Tamil", "அடுத்த தேர்வு எப்போது"),
    ("ui_tamil_refusal.png", "Tamil", "எவ்வளவு வருகை கட்டாயம்"),
]


def ask(page: Page, language: str, question: str) -> None:
    page.get_by_role("group", name="Select language").get_by_title(language).click()
    page.locator("textarea").fill(question)
    page.locator("textarea").press("Enter")
    page.get_by_role("button", name="Play audio").wait_for(timeout=300_000)
    page.get_by_role("button", name="Pipeline trace").click()
    page.wait_for_timeout(400)


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2)
        page.goto(URL)
        page.locator("textarea").wait_for()
        page.wait_for_timeout(800)
        page.screenshot(path=str(HERE / "ui_home.png"))
        for name, language, question in QUESTIONS:
            ask(page, language, question)
            page.screenshot(path=str(HERE / name), full_page=True)
            print(f"saved {name}")
        browser.close()


if __name__ == "__main__":
    main()
