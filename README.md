# SROC

## Straight Rectangles on Canvas

**SROC** (Straight Rectangles on Canvas) is a Python library for manipulating straight rectangles and
handling interactions with a canvas. It provides classes and methods for creating, modifying, and analyzing rectangles,
as well as handling image tiling and canvas scanning operations.

## Features

- **Rect**: A class to represent a rectangle with various utility methods for manipulation and calculation.
- **Rects**: Base class to manage a collection of Rect objects with various utility methods for manipulation and
  calculation.
- **CactusRects**: Specialized class derived from Rects to create and handle collections of adjacent rectangles.
- **RectCanvas**: Base class to represent a canvas with overlayed rectangles and provide various utility methods for
  manipulation and visualization.
- **RectScanner**: Specialized class derived from RectCanvas to scan and detect filled rectangles from the underlying
  canvas and extract as rectangles.
- **CanvasTiler**: Specialized class derived from RectCanvas to handle tiling operations on the canvas.

## Usage

### Rect Class

The `Rect` class represents a rectangle and provides various methods for manipulation and calculation.

```python
from sroc import Rect

# Create a rectangle from a bounding box (x0, y0, x1, y1)
# Same as Rect().fromBox()
rect1 = Rect(box=(0, 0, 100, 100))
# Create a rectangle from a bounding box with specific order
# ('xy' is the default, 'yx, 'xx', 'yy' are supported)
rect1 = Rect(box=(0, 0, 100, 100), order="yx")

# Create a rectangle from tuple of n points (y1, x1, y0, x0)
# Same as Rect().fromPoints(). 
rect2 = Rect(points=((0, 0), (100, 100)))

# Get the width and height of the rectangle
width = rect1.width()
height = rect1.height()

# Rotate the rectangle by 90 degrees
rect1.rotate(90)

# Find distance between rect1 and rect2
distance = rect1.getDistFromRect(rect2, reference='border', type="manhattan")

# Get a tuple with the 4 vertex as y,x points
rect1_points = rect1.to4points(order="yx", sequence="maxmin")

```

#### Viewport

Rectangle viewport can be set both at initialization passing `vp_xmax` and `vp_ymax` or calling
`setViewportLimit(vp_xmax, vp_ymax)`.
Rectangle coordinates can be set or retrieved as fractions of the viewport setting `use_vp_fractions=True`.

### Rects Class

The `Rects` class manages a collection of `Rect` objects and provides methods for manipulation and calculation.

```python
from sroc import Rect
from sroc import Rects

# Create a collection of rectangles
rects1 = Rects()

# Add a rectangle to the collection
rects1.addRect(Rect(box=(0, 0, 100, 100)))

# Add a rectangle from bounding box or tuples of n points (same methods from Rect can be used)
rects2 = Rects().fromBox([(0, 0, 100, 100), (0, 100, 100, 200)])

# Simplify the collection of rectangles (will merge rects to (0, 0, 100, 200))
rects2.simplifyRects(tolerance=5)

# Moves rectangles from rect2 to rect1
rects1.moveRectsFrom(rects2)

# Get a list of the rectangles as 'xy' 'minmax' bounding boxes
rectangles = rects1.toBox()

```

### CactusRects Class

The `CactusRects` class is a specialized class derived from `Rects` to handle specific operations with tolerance and
strategy.

```python
from sroc import CactusRects

# Create a CactusRects object with a seed rectangle
cactus_rects = CactusRects(seedRect=Rect(box=(0, 0, 100, 100)), tolerance=5, strategy="full")

# Add a rectangle to the collection
cactus_rects.addRect(Rect(box=(50, 50, 150, 150)))
```

### RectCanvas Class

The `RectCanvas` class represents a canvas with rectangles and provides various methods for manipulation and
visualization.

```python
from sroc import RectCanvas

# Create a RectCanvas object from an image path
canvas = RectCanvas(image_path="path/to/image.png")

# Get the original canvas
original_canvas = canvas.getOriginalCanvas()

# Get the current canvas with modifications
current_canvas = canvas.getCurrentCanvas(show_rects=True, show_labels=True)
```

### RectScanner Class

The `RectScanner` class is derived from `RectCanvas` to scan and detect filled rectangles in an image.

```python
from sroc.overlay import RectScanner

# Create a RectScanner object from an image path
scanner = RectScanner(image_path="path/to/image.png")

# Scan the image for filled rectangles
filled_rects = scanner.scanFilledRects(color_treshold=200, min_lenght=20)
```

### CanvasTiler Class

The `CanvasTiler` class is derived from `RectCanvas` to handle tiling operations on the canvas.

```python
from sroc.overlay import CanvasTiler

# Create a CanvasTiler object from an image path
tiler = CanvasTiler(image_path="path/to/image.png", tile_width=640, tile_height=640)

# Save the tiles to an output path
tiler.saveTiles(output_path="path/to/output")

# Join the tiles from an input path and save the result
tiler.joinTiles(tiles_path="path/to/tiles", output_path="path/to/output")
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Author

Giovanni Cascione - [ing.cascione@gmail.com](mailto:ing.cascione@gmail.com)
