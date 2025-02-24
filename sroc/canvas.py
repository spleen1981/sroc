"""
Copyright (C) 2025 Giovanni Cascione <ing.cascione@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import concurrent.futures
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .overlay import Rect
from .overlay import Rects


class RectCanvas():
    """
    A class to represent a canvas with rectangles and provide various utility methods for manipulation and visualization.

    Attributes:
    ----------
    rotation : int
        The rotation angle of the canvas.
    originalCanvas : np.ndarray
        The original image canvas.
    image_path : Path
        The path to the image file.
    name : str
        The name of the image file.
    ext : str
        The extension of the image file.
    h_orig : int
        The original height of the image.
    w_orig : int
        The original width of the image.
    w : int
        The current width of the image.
    h : int
        The current height of the image.
    rects : Rects
        A collection of rectangles.
    _rects_labels : list
        Labels for the rectangles.
    _rects_groups : list
        Groups for the rectangles.

    Methods:
    -------
    __init__(self, image_path=None, image_np=None):
        Initializes the RectCanvas object with an image path or numpy array.
    getOriginalCanvas(self, color=False, crop_to_viewport=False, show_rects=False, show_labels=False):
        Returns the original canvas with optional modifications.
    getCurrentCanvas(self, extend=True, color=False, crop_to_viewport=False, show_rects=False, show_labels=False):
        Returns the current canvas with optional modifications.
    originalCanvasSize(self):
        Returns the size of the original canvas.
    currentCanvasSize(self):
        Returns the size of the current canvas.
    moveRectsFrom(self, rects):
        Moves rectangles from another Rects object.
    """

    def __init__(self, image_path=None, image_np=None):
        # TODO double check rotation is actually needed
        self.rotation = 0
        self.originalCanvas = None

        if not image_path is None:
            self.image_path = Path(image_path)
            if not self.image_path.exists():
                raise ValueError(f"File {image_path} does not exist")
            self.name = self.image_path.stem
            self.ext = self.image_path.suffix
        else:
            if not image_np is None:
                self.originalCanvas = image_np
                self.image_path = None
                self.name = 'noname'
                self.ext = 'png'
            else:
                raise ValueError(f"No image provided")

        img = self.getOriginalCanvas()
        self.h_orig, self.w_orig = img.shape[:2]
        self.w, self.h = (self.w_orig, self.h_orig)
        self.rects = Rects()
        self._rects_labels = []
        self._rects_groups = []

    @property
    def rects_groups(self):
        if len(self.rects) > len(self._rects_groups):
            if not len(self._rects_groups):
                next_index = 1
            else:
                next_index = max(self._rects_groups) + 1
            for i in range(len(self._rects_groups), len(self.rects)):
                self._rects_groups.append(next_index)
        return self._rects_groups

    @rects_groups.setter
    def rects_groups(self, groups):
        if not isinstance(groups, list):
            raise ValueError("rects_groups must be a list")
        groups_index = {}
        this_index = 1
        self._rects_groups = []
        for group in groups:
            if group not in groups_index:
                groups_index[group] = this_index
                this_index += 1
            self._rects_groups.append(groups_index[group])

    @property
    def rects_labels(self):
        if len(self.rects) > len(self._rects_labels):
            for i in range(len(self._rects_labels), len(self.rects)):
                self._rects_labels.append("")
        return self._rects_labels

    @rects_labels.setter
    def rects_labels(self, labels):
        if not isinstance(labels, list):
            raise ValueError("rects_labels must be a list")
        self._rects_labels = labels

    def saveCanvas(self, path, original_size=True, color=False, crop_to_viewport=False, show_rects=False, show_labels=False):
        kwargs={'color':color, 'crop_to_viewport':crop_to_viewport, 'show_rects':show_rects, 'show_labels':show_labels}
        print(kwargs)
        if original_size:
            dressed_pic = self.getOriginalCanvas(**kwargs)
        else:
            dressed_pic = self.getOriginalCanvas(**kwargs, extend=True)

        cv2.imwrite(Path(path), dressed_pic)

    def getOriginalCanvas(self, color=False, crop_to_viewport=False, show_rects=False, show_labels=False):
        if not self.originalCanvas is None:
            if color and len(self.originalCanvas.shape) == 2:
                img = cv2.cvtColor(self.originalCanvas, cv2.COLOR_GRAY2BGR)
            elif not color and len(self.originalCanvas.shape) == 3 and self.originalCanvas.shape[2] == 3:
                img = cv2.cvtColor(self.originalCanvas, cv2.COLOR_BGR2GRAY)
            else:
                img = self.originalCanvas
        else:
            if color:
                mode = cv2.IMREAD_COLOR
            else:
                mode = cv2.IMREAD_GRAYSCALE
            img = cv2.imread(self.image_path, mode)

        return self.__rectDress(img, crop_to_viewport, show_rects, show_labels)

    def getCurrentCanvas(self, extend=True, color=False, crop_to_viewport=False, show_rects=False, show_labels=False):
        img = self.getOriginalCanvas(color=color, crop_to_viewport=crop_to_viewport, show_rects=show_rects,
                                     show_labels=show_labels)
        if self.w != self.w_orig or self.h != self.h_orig:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.rotation:
            M = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), self.rotation, 1.0)
            if extend:
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                rot_width = int((self.h * sin) + (self.w * cos))
                rot_height = int((self.h * cos) + (self.w * sin))
            else:
                rot_width = self.w
                rot_height = self.h
            img = cv2.warpAffine(img, M, (rot_width, rot_height))

        return self.__rectDress(img, crop_to_viewport, show_rects, show_labels)

    def __getDressColor(self, index, n):
        colors = plt.colormaps.get_cmap('hsv')
        color = colors(index / n)
        rgb_color = tuple(int(c * 255) for c in color[:3])
        return rgb_color

    def __rectDress(self, img, crop_to_viewport=False, show_rects=False, show_labels=False):
        if show_rects and len(self.rects):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            text_thk = 2
            groups = max(self.rects_groups)
            for i in range(len(self.rects)):
                color = self.__getDressColor(self.rects_groups[i], groups)
                rect = self.rects.getRect(i)
                img = cv2.rectangle(img, rect.topLeft(), rect.bottomRight(), color, 2)

                if show_labels:
                    label_size, base_line = cv2.getTextSize(self.rects_labels[i], font, scale, text_thk)
                    top = max(rect.ymin(), label_size[1] - base_line)
                    left = min(rect.xmin(), img.shape[1] - label_size[0])
                    cv2.rectangle(img, (left, top - label_size[1]),
                                  (left + label_size[0], top + base_line), color, cv2.FILLED)
                    cv2.putText(img, self.rects_labels[i], (left, top), font, scale, (0, 0, 0), text_thk)

        if crop_to_viewport:
            img = img[self.rects.viewPort.ymin():self.rects.viewPort.ymax(),
                  self.rects.viewPort.xmin():self.rects.viewPort.xmax()]
        return img

    def originalCanvasSize(self):
        return self.getOriginalCanvas().shape[:2]

    def currentCanvasSize(self):
        return self.getCurrentCanvas().shape[:2]

    def moveRectsFrom(self, rects):
        self.rects.moveRectsFrom(rects)


class RectScanner(RectCanvas):
    """
    A class derived from RectCanvas to scan and detect filled rectangles in an image.

    Methods:
    -------
    __groupPointsToLines(self, scan_coor):
        Groups points into lines based on the scan coordinate.
    __groupLinesToRects(self, lines, scan_coor):
        Groups lines into rectangles based on the scan coordinate.
    scanFilledRects(self, color_treshold=200, min_lenght=20, adjacent_margin=0, limits_margin=1):
        Scans the image for filled rectangles based on the specified thresholds and margins.
    """

    def __groupPointsToLines(self, scan_coor):
        lines = []
        image = self.getCurrentCanvas()
        height, width = image.shape

        if scan_coor:
            range_out = height
            range_in = width
        else:
            range_out = width
            range_in = height

        for scan_out in range(range_out):
            start = -1
            end = -1
            row = []
            for scan_in in range(range_in):
                if scan_coor:
                    color = image[scan_out, scan_in]
                else:
                    color = image[scan_in, scan_out]

                if color < self.color_treshold:
                    if start == -1:
                        start = scan_in
                        end = -1
                else:
                    if start != -1:
                        end = scan_in
                if end != -1:
                    if end - start > self.length_treshold:
                        row.append([start, end])
                    end = -1
                    start = -1
            if start != -1:
                row.append([start, range_in - 1])

            lines.append(row)

        return lines

    def __groupLinesToRects(self, lines, scan_coor):
        rects = []
        for i in range(1, len(lines)):
            for j in range(len(lines[i])):
                found = False
                for k in range(len(rects)):
                    xmin, ymin, xmax, ymax = rects[k]
                    if scan_coor:
                        scan_dir = ymin
                        margin_min = xmin
                        margin_max = xmax
                    else:
                        scan_dir = ymin
                        margin_min = xmin
                        margin_max = xmax

                    if (i - scan_dir) <= 1 + self.adjacent_margin and abs(
                            lines[i][j][0] - margin_min) <= self.limits_margin and abs(
                        lines[i][j][1] - margin_max) <= self.limits_margin:
                        found = True
                        if scan_coor:
                            rects[k] = [xmin, ymin, xmax, i]
                        else:
                            rects[k] = [xmin, ymin, i, ymax]
                if found == False:
                    if scan_coor:
                        rects.append([lines[i][j][0], i, lines[i][j][1], i])
                    else:
                        rects.append([i, lines[i][j][0], i, lines[i][j][1]])
                found = False
        return rects

    def scanFilledRects(self, color_treshold=200, min_lenght=20, adjacent_margin=0, limits_margin=1):
        self.color_treshold = color_treshold
        self.length_treshold = min_lenght
        self.adjacent_margin = adjacent_margin
        self.limits_margin = limits_margin
        self.rects._moveStdBoxesFrom(
            self.__groupLinesToRects(self.__groupPointsToLines(1), 1) + self.__groupLinesToRects(
                self.__groupPointsToLines(0), 0))
        return self.rects


class CanvasTiler(RectCanvas):
    """
    A class derived from RectCanvas to handle tiling operations on the canvas.

    Attributes:
    ----------
    tile_width : int
        The width of each tile.
    tile_height : int
        The height of each tile.
    tiles_target_no : int
        The target number of tiles.
    __cachedCurrentCanvas : np.ndarray
        Cached current canvas for efficient access.

    Methods:
    -------
    __init__(self, image_path=None, image_np=None, rotation=0, tiles_target_no=64, tile_width=640, tile_height=640, force_horizontal=True):
        Initializes the CanvasTiler object with specified parameters.
    lenX(self):
        Returns the number of tiles along the x-axis.
    lenY(self):
        Returns the number of tiles along the y-axis.
    __len__(self):
        Returns the total number of tiles.
    getTileName(self, index_x, index_y):
        Returns the name of the tile at the specified indices.
    saveTiles(self, output_path):
        Saves the tiles to the specified output path.
    getTile(self, index_x, index_y):
        Returns the tile at the specified indices.
    joinTiles(self, tiles_path, output_path, restore_original_size=False):
        Joins the tiles from the specified path and saves the result.
    tilesCrawler(self, crawler_callback, threads=1, status_callback=None, color=False, results_on_original_canvas=True):
        Crawls through the tiles and applies the specified callback function.
    """

    def __init__(self, image_path=None, image_np=None, rotation=0, tiles_target_no=64, tile_width=640, tile_height=640,
                 force_horizontal=True):
        self.tile_width = tile_width
        self.tile_height = tile_height
        # self.force_horizontal = force_horizontal
        self.tiles_target_no = tiles_target_no
        super().__init__(image_path=image_path, image_np=image_np)
        self.__cachedCurrentCanvas = None
        if self.w % self.tile_width:
            self.w = int(self.w + (self.tile_width - self.w % self.tile_width))
        if self.h % self.tile_height:
            self.h = int(self.h + (self.tile_height - self.h % self.tile_height))

    """
        if self.force_horizontal and self.h > self.w and abs(self.rotation) != 90:
            self.rotation += 90
            self.h, self.w = self.canvasSize()
    """

    def __tileName(self, x, y):
        return f'{self.name}_{x}_{y}{self.ext}'

    """
    def getCanvas(self, extend=True):
        # Increase size to multiples of tile sizes
        def adjustToTileSize():
            if self.w % self.tile_width:
                self.w = int(self.w + (self.tile_width - self.w % self.tile_width))
            if self.h % self.tile_height:
                self.h = int(self.h + (self.tile_height - self.h % self.tile_height))

        adjustToTileSize()

        tiles = self.lenX() + self.lenY()
        warn_tiles = 4 + 2 * self.lenX() + 2 * self.lenY()
        if tiles < self.tiles_target_no:
            delta_tiles = self.tiles_target_no - tiles
            self.w = int(self.w + delta_tiles * self.lenX() / tiles)
            self.h = int(self.h + delta_tiles * self.lenY() / tiles)
            adjustToTileSize()
        elif tiles > warn_tiles:
            warnings.warn(
                f"Number of tiles much bigger than target ({warn_tiles} vs {tiles}). Make sure this is intentional!",
                UserWarning)

        return super().getCanvas(extend)
    """

    def lenX(self):
        return int(self.w / self.tile_width)

    def lenY(self):
        return int(self.h / self.tile_height)

    def __len__(self):
        return self.lenX() * self.lenY()

    def getTileName(self, index_x, index_y):
        return self.__tileName(index_x * self.tile_width, index_y * self.tile_height) + self.ext

    def saveTiles(self, output_path):
        img = self.getCurrentCanvas()

        count = 0
        for x, y in product(range(0, self.w, self.tile_width), range(0, self.h, self.tile_height)):
            box = Rect().fromBox((x, y, x + self.tile_width, y + self.tile_height))
            out = Path(output_path) / self.__tileName(x, y)
            tile = img[box.ymin():box.ymax(), box.xmin():box.xmax()]
            cv2.imwrite(out, tile)
            count += 1
        print(f"Sliced {count} tiles in {str(Path(output_path).resolve())}")

    def __testTileIndex(self, index_x, index_y):
        if not index_x in range(self.lenX()) or not index_y in range(self.lenY()):
            raise ValueError(f"Tile index out of range {index_x},{index_y} vs {self.lenX()},{self.lenY()}")

    def getTile(self, index_x, index_y):
        self.__testTileIndex(index_x, index_y)

        if self.__cachedCurrentCanvas is None:
            canvas = self.getCurrentCanvas()
        else:
            canvas = self.__cachedCurrentCanvas

        return canvas[index_y * self.tile_height:(index_y + 1) * self.tile_height,
               index_x * self.tile_width:(index_x + 1) * self.tile_width]

    def joinTiles(self, tiles_path, output_path, restore_original_size=False):
        tot_x_items = self.lenX()
        tot_y_items = self.lenY()
        new_im = np.zeros((tot_y_items * self.tile_height, tot_x_items * self.tile_width, 3), dtype=np.uint8)
        tiles_path = Path(tiles_path)
        count = 0
        for x, y in product(range(0, self.w, self.tile_width), range(0, self.h, self.tile_height)):
            im = cv2.imread(tiles_path / f"{self.__tileName(x, y)}{self.ext}")
            new_im[y:y + self.tile_height, x:x + self.tile_width] = im
            count += 1

        if restore_original_size == True:
            new_im = cv2.resize(new_im, (self.w_orig, self.h_orig), interpolation=cv2.INTER_LANCZOS4)

        file_out = Path(output_path) / f"{self.name}_joined{self.ext}"

        cv2.imwrite(file_out, new_im)
        print(f"Joined {count} tiles in {str(Path(file_out).resolve())}")

        return str(file_out)

    def tilesCrawler(self, crawler_callback, workers='auto', parallelization_mode='multiprocessing', status_callback=None, color=False,
                     results_on_original_canvas=True):
        self.__cachedCurrentCanvas = self.getCurrentCanvas(color=color)

        # Preparing arguments for the crawler
        crawler_args_list = []
        tiles_map = list(product(range(self.lenX()), range(self.lenY())))
        ratiox = self.w_orig / self.w
        ratioy = self.h_orig / self.h
        for i in range(len(tiles_map)):
            x = tiles_map[i][0]
            y = tiles_map[i][1]
            crawler_args_list.append((self.getTile(x, y), self.tile_width * x, self.tile_height * y, ratiox, ratioy,
                                       crawler_callback, status_callback, results_on_original_canvas, i + 1, len(tiles_map)))

        # Choosing parallel executor
        if workers == 'auto':
            workers = None
        executor = (concurrent.futures.ThreadPoolExecutor(max_workers=workers)
                    if parallelization_mode == "multithreading"
                    else concurrent.futures.ProcessPoolExecutor(max_workers=workers))

        with executor as exec:
            results = list(exec.map(_canvasTilerGlobalCrawler, crawler_args_list))

        combined_results = {}
        if len(results):
            for result in results:
                for key, value in result.items():
                    if key in combined_results:
                        combined_results[key] += value
                    else:
                        combined_results[key] = value

        self.__cachedCurrentCanvas = None
        return combined_results

def _canvasTilerGlobalCrawler(args):
    tile, offset_x, offset_y, ratiox, ratioy, crawler_callback, status_callback, results_on_original_canvas, crawler_counter, total_tiles = args

    if status_callback is not None:
        status_callback(int(crawler_counter/total_tiles*100))
    res = crawler_callback(tile)
    if isinstance(res, dict) and 'offsetBoxes' in res:
        for i in range(len(res['offsetBoxes'])):
            item = res['offsetBoxes'][i]
            rect = Rect().fromBox(item)
            rect.addOffset(offset_x=offset_x, offset_y=offset_y)
            if results_on_original_canvas:
                rect.addScaleFactor(ratiox, ratioy)
            res['offsetBoxes'][i] = rect.toBox()
    return res