import unittest

try:
    from tui.app import AudioModemTUI
except ImportError:
    AudioModemTUI = None


@unittest.skipIf(AudioModemTUI is None, "Textual is not installed")
class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_six_page_smoke(self):
        app = AudioModemTUI()
        async with app.run_test(size=(160, 55)) as pilot:
            await pilot.pause()
            for selector in ("#dashboard", "#encode", "#media", "#decode", "#results", "#profiles"):
                self.assertIsNotNone(app.query_one(selector))
            self.assertIsNotNone(app.query_one("#pipeline-start"))
