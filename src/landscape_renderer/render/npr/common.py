import os

_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
          "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
FONT = next((f for f in _FONTS if os.path.exists(f)), _FONTS[0])
STYLE_RU = {"pero": "перо", "karandash": "карандаш", "pastel": "пастель", "akvarel": "акварель",
            "pero_akvarel": "перо + акварель", "pero_marker": "перо + маркер"}
