# Validation Sample Images

Place optional local benchmark images here:

```text
tests/validation_samples/
  valid/
    aerial-city.png
    farmland.jpg
    coastline.png
  invalid/
    id-card.jpg
    indoor-room.png
    screenshot.png
```

Run:

```bash
python validation_benchmark.py
```

The script also includes built-in synthetic benchmark cases, so this folder can stay empty. Do not commit copyrighted, private, or sensitive images. Use public-domain/open-license samples or keep local-only files untracked.
