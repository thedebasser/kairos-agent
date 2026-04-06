# Asset Sourcing Guide

All 3D assets, HDRIs, and textures are sourced from [Poly Haven](https://polyhaven.com/) (CC0 licensed) using the `polydown` CLI tool.

## Setup

```bash
pip install polydown
```

## Download Commands

### Models (~100-130 models)

```bash
# Furniture — tables, chairs, shelves, beds (~30-40 models)
polydown models -c furniture -f kairos-assets/models -s 1k -no

# Props — plates, cups, bottles, books, food, tools (~50-60 models)
polydown models -c props -f kairos-assets/models -s 1k -no

# Decorative — vases, picture frames, candles, clocks (~20-30 models)
polydown models -c decorative -f kairos-assets/models -s 1k -no
```

`-s 1k` keeps textures at 1K resolution (small file sizes, plenty for renders). `-no` skips downloading preview images.

Each model is ~5-20MB at 1K textures, so the whole set is around 1-2GB.

### HDRIs (outdoor skyboxes)

```bash
# Grab 10 HDRIs for outdoor scenes
polydown hdris -f kairos-assets/hdris -s 2k -it 10
```

`-it 10` limits to 10 HDRIs.

### Floor/Ground Textures

```bash
# Floor textures — wood, concrete, tile, etc.
polydown textures -c floor -f kairos-assets/textures -s 1k -no
```

## File Locations

After downloading, assets are placed in the repo under:

```
assets/
├── models/     # .blend files (furniture, props, decorative)
├── hdris/      # .hdr files (2K outdoor skyboxes)
└── textures/   # Floor/ground textures (1K)
```

## Notes

- All assets are CC0 licensed — no attribution required.
- Assets are git-ignored (binary files, ~1-2GB total). See `.gitignore`.
- The asset catalogue generator (`scripts/generate_asset_catalogue.py`) scans these directories to produce the machine-readable catalogue YAML.
