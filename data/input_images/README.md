# Input Images for 2D→3D Pipeline

Place your fish images (or any images) here to process through the pipeline.

## Usage

Run the pipeline with:
```bash
python -m fast_cody.apps.pipeline --image data/input_images/your_fish_image.png
```

Or use absolute paths:
```bash
python -m fast_cody.apps.pipeline --image /path/to/your/image.png
```

## Supported Formats
- PNG, JPG, JPEG (any format supported by the Hunyuan API)

## Output
Processed models will be saved to:
- `outputs/hunyuan/` - Hunyuan-generated GLB/OBJ files
- `outputs/converted/` - Converted OBJ, texture, and MSH files (timestamped folders)
