"""Optional live UI smoke test: pip install playwright; use an installed Chrome."""
from pathlib import Path
import argparse
import base64
import shutil


def main():
    from playwright.sync_api import sync_playwright, expect
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    outputs = root / 'outputs' / 'ui'
    outputs.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which('google-chrome'), headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        response = page.goto(args.url, wait_until='networkidle')
        assert response.status == 200
        assert page.title() == 'MedRestore · Image restoration'
        assert page.locator('#restoreButton').is_disabled()
        page.screenshot(path=str(outputs / 'desktop-empty.png'), full_page=True)
        page.get_by_role('button', name='How it works').click()
        assert page.locator('#helpDialog').is_visible()
        page.get_by_role('button', name='Got it', exact=True).click()
        assert not page.locator('#helpDialog').is_visible()
        page.get_by_role('button', name='Try a sample X-ray').click()
        expect(page.locator("#restoreButton")).to_be_enabled(timeout=30000)
        assert page.locator('#fileName').inner_text() == 'sample-xray.png'
        page.get_by_role('button', name='Restore image', exact=True).click()
        expect(page.locator("#downloadButton")).to_be_enabled(timeout=30000)
        assert page.locator('#statusTitle').inner_text() == 'Restoration complete'
        assert page.get_by_role('button', name='Compare', exact=True).get_attribute('aria-pressed') == 'true'
        assert page.locator('#comparisonOverlay').is_visible()
        slider = page.get_by_role('slider', name='Before and after comparison')
        slider.focus()
        slider.press('ArrowRight')
        assert slider.input_value() == '51'
        assert page.locator('#imageCanvas').evaluate('(el) => el.style.getPropertyValue("--split")') == '51%'
        page.get_by_role('button', name='Restored', exact=True).click()
        assert not page.locator('#comparisonOverlay').is_visible()
        page.get_by_role('button', name='Original', exact=True).click()
        assert page.locator('#baseImage').get_attribute('alt') == 'Original uploaded image'
        page.get_by_role('button', name='Compare', exact=True).click()
        with page.expect_download() as info:
            page.get_by_role('button', name='Download PNG', exact=True).click()
        download = info.value
        assert download.suggested_filename == 'sample-xray_restored.png'
        download.save_as(str(outputs / download.suggested_filename))
        # Strength changes must be applied explicitly, never silently relabel an old result.
        strength = page.locator('#restorationStrength')
        strength.focus()
        strength.press('ArrowLeft')
        assert page.locator('#strengthValue').inner_text() == '45%'
        page.get_by_role('button', name='Update preview', exact=True).click()
        expect(page.locator('#downloadButton')).to_be_enabled(timeout=30000)
        assert '45% strength' in page.locator('#statusDetail').inner_text()
        page.screenshot(path=str(outputs / 'desktop-result.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(outputs / 'mobile-result.png'), full_page=True)
        page.get_by_role('button', name='Remove image').click()
        assert page.locator('#dropZone').is_visible()
        assert page.locator('#restoreButton').is_disabled()
        assert page.locator('#downloadButton').is_disabled()
        page.locator('#imageUpload').set_input_files({'name': 'invalid.png', 'mimeType': 'image/png', 'buffer': b'invalid image'})
        page.wait_for_selector('#errorNotice:visible')
        assert 'valid PNG' in page.locator('#errorNotice').inner_text()
        # A real file upload recovers from the failed upload without refreshing.
        page.locator('#imageUpload').set_input_files(str(root / 'web/static/sample.png'))
        expect(page.locator("#restoreButton")).to_be_enabled(timeout=30000)
        assert not page.locator('#errorNotice').is_visible()
        page.get_by_role('button', name='Remove image').click()
        # Exercise the actual drag/drop listener as well as the file chooser.
        content = base64.b64encode((root / 'web/static/sample.png').read_bytes()).decode()
        page.evaluate('''(content) => {
            const bytes = Uint8Array.from(atob(content), (c) => c.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'dropped-xray.png', {type: 'image/png'}));
            document.querySelector('#dropZone').dispatchEvent(new DragEvent('drop', {dataTransfer: transfer, bubbles: true, cancelable: true}));
        }''', content)
        expect(page.locator("#restoreButton")).to_be_enabled(timeout=30000)
        assert page.locator('#fileName').inner_text() == 'dropped-xray.png'
        assert not errors, errors
        print('PASS: sample, upload, drag/drop, restoration, comparison modes, keyboard slider, PNG download, reset, error recovery, guide and mobile layout')
        print(f'Screenshots saved to {outputs}')
        browser.close()


if __name__ == '__main__':
    main()
