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
import math
import numbers
from functools import lru_cache

class Rect:
    def __init__(self, points=None, box=None, order='xy', xref=None, yref=None, fractions=False):
        self.__xref = xref
        self.__yref = yref
        self.__xmin = 0
        self.__xmax = 0
        self.__ymin = 0
        self.__ymax = 0
        if box:
            self.fromBox(box, order, xref, yref, fractions)
        elif points:
            self.fromPoints(points, order, xref, yref, fractions)
        else:
            self.fromBox((0, 0, 0, 0))

    def __str__(self):
        return f"SRect (top_left:[{self.xmin()}, {self.ymin()}] bottom_right:[{self.xmax()}, {self.ymax()}] size:[{self.width()}, {self.height()}] viewport:[{self._xref}, {self._yref}])"

    def __hash__(self):
        return hash((self.__xmin, self.__xmax, self.__ymin, self.__ymax, self.__xref, self.__yref))

    def __eq__(self, other):
        if isinstance(other, Rect):
            return (self.__xmin, self.__xmax, self.__ymin, self.__ymax, self.__xref, self.__yref) == (
                other.__xmin, other.__xmax, other.__ymin, other.__ymax, other.__xref, other.__yref)
        return False

    def __setExtremes(self, x, y, xref=None, yref=None, fractions=False):
        xmax, ymax = max(x), max(y)
        xref = xref or (xmax if xmax else 1)
        yref = yref or (ymax if ymax else 1)
        self._xref, self._yref = xref, yref
        if fractions:
            xref, yref = 1, 1
        self._xmin = min(x) / xref
        self._ymin = min(y) / yref
        self._xmax = xmax / xref
        self._ymax = ymax / yref

    def scaleReference(self, xref, yref):
        self._xmin = min(1, self._xmin * self._xref / xref)
        self._xmax = min(1, self._xmax * self._xref / xref)
        self._ymin = min(1, self._ymin * self._yref / yref)
        self._ymax = min(1, self._ymax * self._yref / yref)
        self._xref, self._yref = xref, yref

    def setReference(self, xref, yref):
        self._xref = xref
        self._yref = yref

    def fromBox(self, box=(0, 0, 0, 0), order='xy', xref=None, yref=None, fractions=False):
        if order == 'xy':
            x, y = (box[0], box[2]), (box[1], box[3])
        elif order == 'yx':
            x, y = (box[1], box[3]), (box[0], box[2])
        elif order == 'xx':
            x, y = (box[0], box[1]), (box[2], box[3])
        elif order == 'yy':
            x, y = (box[2], box[3]), (box[0], box[1])

        self.__setExtremes(x, y, xref, yref, fractions)
        return self

    def fromPoints(self, points=((0, 0), (0, 0), (0, 0), (0, 0)), order='xy', xref=None, yref=None, fractions=False):
        index_x = order == 'yx'
        x = tuple(point[index_x] for point in points)
        y = tuple(point[not index_x] for point in points)
        self.__setExtremes(x, y, xref, yref, fractions)
        return self

    @lru_cache(maxsize=None)
    def width(self):
        return abs(self.xmax() - self.xmin())

    @lru_cache(maxsize=None)
    def height(self):
        return abs(self.ymax() - self.ymin())

    @lru_cache(maxsize=None)
    def area(self):
        return self.width() * self.height()

    @lru_cache(maxsize=None)
    def perimeter(self):
        return 2 * (self.width() + self.height())

    @lru_cache(maxsize=None)
    def xmin(self, xref=None):
        xref = xref or self._xref
        return int(self.__xmin * xref)

    @lru_cache(maxsize=None)
    def xmax(self, xref=None):
        xref = xref or self._xref
        return int(self.__xmax * xref)

    @lru_cache(maxsize=None)
    def ymin(self, yref=None):
        yref = yref or self._yref
        return int(self.__ymin * yref)

    @lru_cache(maxsize=None)
    def ymax(self, yref=None):
        yref = yref or self._yref
        return int(self.__ymax * yref)

    def rotate(self, rotation, xr=0, yr=0, fractions=False):
        if rotation % 90:
            raise ValueError("Multiple of 90° are only accepted for straight rectangles")

        rad = rotation * math.pi / 180
        if not fractions:
            xr /= self._xref
            yr /= self._yref

        def rotate_x(x, y):
            return (x - xr) * math.cos(rad) - (y - yr) * math.sin(rad) + xr

        def rotate_y(x, y):
            return (x - xr) * math.sin(rad) + (y - yr) * math.cos(rad) + yr

        x = (rotate_x(self.__xmin, self.__ymin), rotate_x(self.__xmax, self.__ymax))
        y = (rotate_y(self.__xmin, self.__ymin), rotate_y(self.__xmax, self.__ymax))

        self.__setExtremes(x, y, self._xref, self._yref, True)
        return self

    def addOffset(self, offset_x=0, offset_y=0, xref=None, yref=None, fractions=False):
        xmin, xmax = (self.__xmin, self.__xmax) if fractions else (self.xmin(), self.xmax())
        ymin, ymax = (self.__ymin, self.__ymax) if fractions else (self.ymin(), self.ymax())
        self.__setExtremes((xmin + offset_x, xmax + offset_x), (ymin + offset_y, ymax + offset_y), xref, yref,
                           fractions)
        return self

    def addScaleFactor(self, factor_x=1, factor_y=1, xref=None, yref=None, fractions=False):
        xmin, xmax = (self.__xmin, self.__xmax) if fractions else (self.xmin(), self.xmax())
        ymin, ymax = (self.__ymin, self.__ymax) if fractions else (self.ymin(), self.ymax())
        self.__setExtremes((xmin * factor_x, xmax * factor_x), (ymin * factor_y, ymax * factor_y), xref, yref,
                           fractions)
        return self

    def addBorder(self, borderx=0, bordery=0, fractions=False, expand=True):
        _borderx, _bordery = (borderx / self._xref, bordery / self._yref) if not fractions else (borderx, bordery)
        borderx, bordery = (self._xref * _borderx, self._yref * _bordery) if fractions else (borderx, bordery)

        if self.xmin() < borderx and expand:
            self.__xmin = 0
            self.scaleReference(self._xref + borderx - self.xmin(), self._yref)
        else:
            self.__xmin = max(0, self.__xmin - _borderx)

        if borderx > (self._xref - self.xmax()) and expand:
            self.scaleReference(borderx + self.xmax(), self._yref)
            self.__xmax = 1
        else:
            self.__xmax = min(1, self.__xmax + _borderx)

        if self.ymin() < bordery and expand:
            self.__ymin = 0
            self.scaleReference(self._xref, self._yref + bordery - self.ymin())
        else:
            self.__ymin = max(0, self.__ymin - _bordery)

        if bordery > self._yref - self.ymax() and expand:
            self.scaleReference(self._xref, bordery + self.ymax())
            self.__ymax = 1
        else:
            self.__ymax = min(1, self.__ymax + _bordery)

    def toBox(self, order='xy', sequence='minmax', fractions=False):
        xref, yref = (1, 1) if fractions else (self.xref, self.yref)
        coords = (self.xmin(xref), self.ymin(yref), self.xmax(xref), self.ymax(yref))

        if sequence == "maxmin":
            coords = coords[::-1]

        if order == 'yx':
            coords = coords[1], coords[0], coords[3], coords[2]
        elif order == 'xx':
            coords = coords[0], coords[2], coords[1], coords[3]
        elif order == 'yy':
            coords = coords[2], coords[3], coords[0], coords[1]

        return coords

    def to2Points(self, order='xy', sequence='minmax', fractions=False):
        c1, c2, c3, c4 = self.toBox(order, sequence, fractions)
        return (c1, c2), (c3, c4)

    def to4Points(self, order='xy', sequence='minmax', fractions=False):
        c1, c2, c3, c4 = self.toBox(order, sequence, fractions)
        return (c1, c2), (c3, c4), (c1, c4), (c3, c2)

    @property
    def xref(self):
        return self.__xref

    @xref.setter
    def xref(self, value):
        self.__xref = value
        self.invalidate_cache()

    @property
    def yref(self):
        return self.__yref

    @yref.setter
    def yref(self, value):
        self.__yref = value
        self.invalidate_cache()

    @property
    def _xmin(self):
        return self.__xmin

    @_xmin.setter
    def _xmin(self, value):
        self.__xmin = value
        self.invalidate_cache()

    @property
    def _xmax(self):
        return self.__xmax

    @_xmax.setter
    def _xmax(self, value):
        self.__xmax = value
        self.invalidate_cache()

    @property
    def _ymin(self):
        return self.__ymin

    @_ymin.setter
    def _ymin(self, value):
        self.__ymin = value
        self.invalidate_cache()

    @property
    def _ymax(self):
        return self.__ymax

    @_ymax.setter
    def _ymax(self, value):
        self.__ymax = value
        self.invalidate_cache()

    def invalidate_cache(self):
        self.xmax.cache_clear()
        self.xmin.cache_clear()
        self.ymax.cache_clear()
        self.ymin.cache_clear()
        self.width.cache_clear()
        self.height.cache_clear()
        self.area.cache_clear()
        self.perimeter.cache_clear()
        self.center.cache_clear()
        self.getDistFromRect.cache_clear()

    @lru_cache(maxsize=None)
    def center(self, order='xy'):
        return self.__pointProvider((self.xmin() + self.xmax()) / 2, (self.ymin() + self.ymax()) / 2, order)

    def topLeft(self, order='xy'):
        return self.__pointProvider(self.xmin(), self.ymin(), order)

    def bottomRight(self, order='xy'):
        return self.__pointProvider(self.xmax(), self.ymax(), order)

    def __pointProvider(self, x, y, order):
        return (int(x), int(y)) if order == 'xy' else (int(y), int(x))

    def __pointsDistance(self, x0=0, y0=0, x1=0, y1=0, type="cartesian"):
        if type == "cartesian":
            return int(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        elif type == "cartesian_squares":
            return (x1 - x0) ** 2 + (y1 - y0) ** 2
        elif type == "manhattan":
            return abs(x1 - x0) + abs(y1 - y0)
        else:
            raise ValueError("Unknown distance type specified")

    def union(self, rect):
        x = (min(self.xmin(), rect.xmin()), max(self.xmax(), rect.xmax()))
        y = (min(self.ymin(), rect.ymin()), max(self.ymax(), rect.ymax()))
        self.__setExtremes(x, y, max(self._xref, rect.xref), max(self._yref, rect.yref))
        return self

    @lru_cache(maxsize=None)
    def getDistFromRect(self, rect, reference="border", type="cartesian"):
        if reference == 'center':
            x0, y0 = rect.center()
            x1, y1 = self.center()
        elif reference == 'border':
            x0, y0 = 0, 0
            x1 = rect.xmin() - self.xmax() if self.xmax() < rect.xmin() else self.xmin() - rect.xmax() if self.xmin() > rect.xmax() else 0
            y1 = rect.ymin() - self.ymax() if self.ymax() < rect.ymin() else self.ymin() - rect.ymax() if self.ymin() > rect.ymax() else 0
        else:
            raise ValueError("Invalid reference to calculate distance")
        return (x1, y1), self.__pointsDistance(x0, y0, x1, y1, type)


class Rects:
    def __init__(self):
        self.viewPort = None
        self.tempRect = Rect()
        self.inboundMapping = {
            'fromBox': 'fromBox',
            'fromPoints': 'fromPoints',
        }
        self.outboundMapping = {
            'toBox': 'toBox',
            'to2Points': 'to2Points',
            'to4Points': 'to4Points',
        }
        self.rects = []
        # self._rects_lookup = {'x': {'min': [], 'rects_idx': []}, 'y': {'min': [], 'rects_idx': []}}

    def __getattr__(self, method):
        if method in self.inboundMapping:
            inboundMethod = getattr(self.tempRect, self.inboundMapping[method])

            def wrapper(*args, **kwargs):
                if args:
                    rects = args[0]
                    args = args[1:]
                    for rect in rects:
                        self.addRect(inboundMethod(rect, *args, **kwargs), update_viewport=False)
                    self.updateViewport()
                    return self
                raise AttributeError(
                    f"'{self.__class__.__name__}'Invalid arguments, list of rectangles is to be provided.")

            return wrapper
        elif method in self.outboundMapping:
            outboundMethod = getattr(self.tempRect, self.outboundMapping[method])

            def wrapper(*args, **kwargs):
                res = []
                for rect in self.rects:
                    self.tempRect.fromBox(rect)
                    res.append(outboundMethod(*args, **kwargs))
                return res

            return wrapper
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{method}'")

    def __str__(self):
        return f"SRects ({len(self.rects)} Rects, overall top_left:[{self.viewPort.xmin()}, {self.viewPort.ymin()}] overall bottom_right:[{self.viewPort.xmax()}, {self.viewPort.ymax()}] overall size:[{self.viewPort.width()}, {self.viewPort.height()}] viewport:[{self.viewPort.xref}, {self.viewPort.yref}])"

    def __len__(self):
        return len(self.rects)

    def updateViewport(self):
        if len(self.rects):
            self.viewPort = Rect(box=(min([rect[0] for rect in self.rects]), min([rect[1] for rect in self.rects]),
                                      max([rect[2] for rect in self.rects]), max([rect[3] for rect in self.rects])))

    def addRect(self, rect, update_viewport=True):
        self.rects.append((rect.toBox()))

        if update_viewport:
            # extend mainRect
            if self.viewPort is None:
                self.viewPort = Rect().fromBox(rect.toBox())
            else:
                self.viewPort.union(rect)
        return True

    """
    def add_lookup(self, rect, idx=None):
        def _get_target_idx(coor, coor_min):
            res=None
            len_list=len(self._rects_lookup[coor]['rects_idx'])
            for i in range(len_list):
                if coor_min<self._rects_lookup[coor]['min'][i]:
                    res=i
                    break
            if res is None: res=len_list
            return len_list

        def _add_to_lookup(coor, coor_min, idx):
            target_idx=_get_target_idx(coor, coor_min)
            if idx is None:
                idx= len(self.rects)-1
            self._rects_lookup[coor]['min'].insert(target_idx, coor_min)
            self._rects_lookup[coor]['rects_idx'].insert(target_idx, idx)

        _add_to_lookup('x',rect.xmin(), idx)
        _add_to_lookup('y', rect.ymin(), idx)
    """

    def simplifyRects(self, tolerance=5):
        if len(self.rects):
            rect_new = [self.rects.pop(0)]
        else:
            rect_new = []
        square_tolerance = tolerance ** 2

        def can_merge(test_coor_idx, align_coor_idx):
            test_align_min = (rect_new[-1][test_coor_idx[0]] - this_rect[test_coor_idx[0]]) ** 2 < square_tolerance
            test_align_max = (rect_new[-1][test_coor_idx[1]] - this_rect[test_coor_idx[1]]) ** 2 < square_tolerance
            if test_align_min and test_align_max:
                test_margin1 = (rect_new[-1][align_coor_idx[0]] - this_rect[align_coor_idx[1]]) ** 2 < square_tolerance
                test_margin2 = (rect_new[-1][align_coor_idx[1]] - this_rect[align_coor_idx[0]]) ** 2 < square_tolerance
                if test_margin1 or test_margin2:
                    return True
            return False

        while len(self.rects):
            reset = True
            while reset == True:
                reset = False
                for j in range(len(self.rects) - 1, -1, -1):
                    this_rect = self.rects.pop(j)
                    if not can_merge((1, 3), (0, 2)) and not can_merge((0, 2), (1, 3)):
                        rect_new.append(this_rect)
                    else:
                        rect_new[-1] = [min(this_rect[0], rect_new[-1][0]), min(this_rect[1], rect_new[-1][1]),
                                        max(this_rect[2], rect_new[-1][2]), max(this_rect[3], rect_new[-1][3])]
                        reset = True
                        break
        self.rects = rect_new
        self.updateViewport()
        return self

    def moveRectsFrom(self, rects):
        startLen = len(rects)
        i = 0
        for i in range(len(rects) - 1, -1, -1):
            if self.addRect(Rect().fromBox(rects.rects[i]), update_viewport=False):
                rects.rects.pop(i)

        if startLen > len(rects):
            self.updateViewport()
            return True
        return False

    def __getRect(self, index, then_remove=False):
        # if index in range(len(self)):
        rect = Rect().fromBox(self.rects[index])
        if then_remove:
            del (self.rects[index])
            self.updateViewport()
        return rect

    def getRect(self, index):
        return self.__getRect(index, then_remove=False)

    def popRect(self, index):
        return self.__getRect(index, then_remove=True)


class CactusRects(Rects):
    def __init__(self, seedRect, tolerance=5, strategy="full"):
        super().__init__()
        super().addRect(seedRect)
        # self.seed = seedRect
        self.tolerance = tolerance
        # self.roi_size = 50
        self.strategy = strategy

    def addRect(self, rect, update_viewport=True):
        square_tolerance = self.tolerance ** 2
        # See if new rect is close enough to global boundaries
        dir, dist = self.viewPort.getDistFromRect(rect, reference="border", type="cartesian_squares")
        # print(f"self.mainRect {self.mainRect} rect {rect} dist {dist}")
        if dist > square_tolerance:
            return False
        """
        # Exclude internal boxes depending on external rect approach direction
        boundary_rects = []
        boundary_rects_x = []

        xmin=rect.xmin()
        xmin_min=xmin-(self.tolerance+self.roi_size)
        xmin_max=xmin+(self.tolerance + self.roi_size)
        ymin=rect.ymin()
        ymin_min=ymin-(self.tolerance+self.roi_size)
        ymin_max=ymin+(self.tolerance+self.roi_size)

        for i in range(len(self._rects_lookup['x']['min'])):
            this_idx=self._rects_lookup['y']['rects_idx'][i]
            this_coormin=self._rects_lookup['y']['min'][i]
            if not xmin_min > this_coormin and not xmin_max < this_coormin:
                boundary_rects_x.append(this_idx)

        for i in range(len(self._rects_lookup['y']['min'])):
            this_idx=self._rects_lookup['y']['rects_idx'][i]
            this_coormin=self._rects_lookup['y']['min'][i]
            if not ymin_min > this_coormin and not ymin_max < this_coormin and this_idx in boundary_rects_x:
                boundary_rects.append(self.rects[this_idx])


        """

        # print(boundary_rects)
        boundary_rects = []
        xmin = self.viewPort.xmin()
        ymin = self.viewPort.ymin()
        xmax = self.viewPort.xmax()
        ymax = self.viewPort.ymax()

        if self.strategy == 'full':
            boundary_rects = [r for r in self.rects]
        elif self.strategy == 'boundaries_only':
            for r in self.rects:
                if r[0] == xmin or r[1] == ymin or r[2] == xmax or r[3] == ymax:
                    boundary_rects.append(r)
        else:
            raise ValueError("Unknown merge strategy")

        """
        print(dir)
        print(dist)
        for r in self.rects:
            if dir[0] > 0 and r[2] == xmax or dir[0] < 0 and r[0] == xmin:
                boundary_rects.append(r)
                continue

            if dir[1] > 0 and r[1] == ymin or dir[1] < 0 and r[3] == ymax:
                boundary_rects.append(r)
                continue
        """

        for i in range(len(boundary_rects) - 1, -1, -1):
            _, rect_dist = rect.getDistFromRect(self.tempRect.fromBox(boundary_rects[i]), reference="border",
                                                type="cartesian_squares")
            if rect_dist <= square_tolerance:
                super().addRect(rect, update_viewport=update_viewport)
                return True
        return False

    def moveRectsFrom(self, rects):
        while super().moveRectsFrom(rects):
            pass

    def __getattr__(self, method):
        if method in self.inboundMapping:
            def wrapper(*args, **kwargs):
                tempRects = Rects()
                inboundMethod = getattr(tempRects, self.inboundMapping[method])
                inboundMethod(*args, **kwargs)
                self.moveRectsFrom(tempRects)

            return wrapper
        else:
            return super().__getattr__(method)
