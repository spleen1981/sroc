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


class Rect:
    """
    A class to represent a rectangle with various utility methods for manipulation and calculation.

    Attributes:
    ----------
    _viewPortLimit : list
        The viewport limits (vp_xmax, vp_ymax).
    _innerRect : list
        The inner rectangle coordinates (xmin, ymin, xmax, ymax).

    Methods:
    -------
    __init__(self, points=None, box=None, order='xy', vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        Initializes the Rect object with either points or a bounding box.
    __str__(self):
        Returns a string representation of the Rect object.
    __hash__(self):
        Returns the hash value of the Rect object.
    __eq__(self, other):
        Checks if two Rect objects are equal.
    __setExtremes(self, x, y, vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        Sets the extreme coordinates of the rectangle.
    scaleViewportLimit(self, vp_xmax, vp_ymax):
        Scales the viewport limits.
    setViewportLimit(self, vp_xmax, vp_ymax):
        Sets the viewport limits.
    getViewPortLimit(self):
        Returns the viewport limits.
    fromBox(self, box=(0, 0, 0, 0), order='xy', vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        Initializes the rectangle from a bounding box.
    fromPoints(self, points=((0, 0), (0, 0), (0, 0), (0, 0)), order='xy', vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        Initializes the rectangle from a set of points.
    width(self):
        Returns the width of the rectangle.
    height(self):
        Returns the height of the rectangle.
    area(self):
        Returns the area of the rectangle.
    perimeter(self):
        Returns the perimeter of the rectangle.
    xmin(self, vp_xmax=None):
        Returns the minimum x-coordinate.
    xmax(self, vp_xmax=None):
        Returns the maximum x-coordinate.
    ymin(self, vp_ymax=None):
        Returns the minimum y-coordinate.
    ymax(self, vp_ymax=None):
        Returns the maximum y-coordinate.
    rotate(self, rotation, xr=0, yr=0, use_vp_fractions=False):
        Rotates the rectangle by a specified angle.
    addOffset(self, offset_x=0, offset_y=0, vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        Adds an offset to the rectangle coordinates.
    addScaleFactor(self, factor_x=1, factor_y=1, vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        Scales the rectangle by a specified factor.
    addBorder(self, borderx=0, bordery=0, use_vp_fractions=False, expand=True):
        Adds a border to the rectangle.
    toBox(self, order='xy', sequence='minmax', use_vp_fractions=False):
        Returns the rectangle as a bounding box.
    to2Points(self, order='xy', sequence='minmax', use_vp_fractions=False):
        Returns the rectangle as two points.
    to4Points(self, order='xy', sequence='minmax', use_vp_fractions=False):
        Returns the rectangle as four points.
    center(self, order='xy'):
        Returns the center point of the rectangle.
    topLeft(self, order='xy'):
        Returns the top-left point of the rectangle.
    bottomRight(self, order='xy'):
        Returns the bottom-right point of the rectangle.
    __pointProvider(self, x, y, order):
        Provides a point in the specified order.
    __pointsDistance(self, x0=0, y0=0, x1=0, y1=0, type="cartesian"):
        Calculates the distance between two points.
    union(self, rect):
        Unites the current rectangle with another rectangle.
    getDistFromRect(self, rect, reference="border", type="cartesian"):
        Calculates the distance from another rectangle.
    _getDistFromStdBox(self, box, reference="border", type="cartesian"):
        Calculates the distance from a standard box.
    """

    def __init__(self, points=None, box=None, order='xy', vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        self._viewPortLimit = [0, 0]  # (vp_xmax, vp_ymax)
        self._innerRect = [0, 0, 0, 0]  # (xmin, ymin, xmax, ymax)
        if box:
            self.fromBox(box, order, vp_xmax, vp_ymax, use_vp_fractions)
        elif points:
            self.fromPoints(points, order, vp_xmax, vp_ymax, use_vp_fractions)
        else:
            self.fromBox((0, 0, 0, 0))

    def __str__(self):
        return f"SRect (top_left:[{self.xmin()}, {self.ymin()}] bottom_right:[{self.xmax()}, {self.ymax()}] size:[{self.width()}, {self.height()}] viewport:[{self._viewPortLimit[0]}, {self._viewPortLimit[1]}])"

    def __hash__(self):
        return hash((self._innerRect, self._viewPortLimit))

    def __eq__(self, other):
        if isinstance(other, Rect):
            return (self._innerRect, self._viewPortLimit) == (other._innerRect, other._viewPortLimit)
        return False

    def __setExtremes(self, x, y, vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        xmax, ymax = max(x), max(y)
        vp_xmax = vp_xmax or (xmax if xmax else 1)
        vp_ymax = vp_ymax or (ymax if ymax else 1)
        self._viewPortLimit = (vp_xmax, vp_ymax)
        if use_vp_fractions:
            vp_xmax, vp_ymax = 1, 1
        self._innerRect = (min(x) / vp_xmax, min(y) / vp_ymax, xmax / vp_xmax, ymax / vp_ymax)

    def changeViewportLimits(self, vp_xmax, vp_ymax, scale_inner_rect=True):
        xmin, ymin, xmax, ymax = self._innerRect
        self._viewPortLimit = (vp_xmax, vp_ymax)
        if scale_inner_rect:
            self._innerRect = (
                min(1, xmin * self._viewPortLimit[0] / vp_xmax), min(1, ymin * self._viewPortLimit[1] / vp_ymax),
                min(1, xmax * self._viewPortLimit[0] / vp_xmax), min(1, ymax * self._viewPortLimit[1] / vp_ymax))

    def getViewPortLimit(self):
        return self._viewPortLimit

    def fromBox(self, box=(0, 0, 0, 0), order='xy', vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        if order == 'xy':
            x, y = (box[0], box[2]), (box[1], box[3])
        elif order == 'yx':
            x, y = (box[1], box[3]), (box[0], box[2])
        elif order == 'xx':
            x, y = (box[0], box[1]), (box[2], box[3])
        elif order == 'yy':
            x, y = (box[2], box[3]), (box[0], box[1])

        self.__setExtremes(x, y, vp_xmax, vp_ymax, use_vp_fractions)
        return self

    def fromPoints(self, points=((0, 0), (0, 0), (0, 0), (0, 0)), order='xy', vp_xmax=None, vp_ymax=None,
                   use_vp_fractions=False):
        index_x = order == 'yx'
        x = tuple(point[index_x] for point in points)
        y = tuple(point[not index_x] for point in points)
        self.__setExtremes(x, y, vp_xmax, vp_ymax, use_vp_fractions)
        return self

    def width(self):
        return abs(self.xmax() - self.xmin())

    def height(self):
        return abs(self.ymax() - self.ymin())

    def area(self):
        return self.width() * self.height()

    def perimeter(self):
        return 2 * (self.width() + self.height())

    def xmin(self, vp_xmax=None):
        vp_xmax = vp_xmax or self._viewPortLimit[0]
        return int(self._innerRect[0] * vp_xmax)

    def xmax(self, vp_xmax=None):
        vp_xmax = vp_xmax or self._viewPortLimit[0]
        return int(self._innerRect[2] * vp_xmax)

    def ymin(self, vp_ymax=None):
        vp_ymax = vp_ymax or self._viewPortLimit[1]
        return int(self._innerRect[1] * vp_ymax)

    def ymax(self, vp_ymax=None):
        vp_ymax = vp_ymax or self._viewPortLimit[1]
        return int(self._innerRect[3] * vp_ymax)

    def rotate(self, rotation, xr=0, yr=0, use_vp_fractions=False):
        if rotation % 90:
            raise ValueError("Multiple of 90° are only accepted for straight rectangles")

        rad = rotation * math.pi / 180
        if not use_vp_fractions:
            xr /= self._viewPortLimit[0]
            yr /= self._viewPortLimit[1]

        def rotate_x(x, y):
            return (x - xr) * math.cos(rad) - (y - yr) * math.sin(rad) + xr

        def rotate_y(x, y):
            return (x - xr) * math.sin(rad) + (y - yr) * math.cos(rad) + yr

        x = (rotate_x(self._innerRect[0], self._innerRect[1]), rotate_x(self._innerRect[2], self._innerRect[3]))
        y = (rotate_y(self._innerRect[0], self._innerRect[1]), rotate_y(self._innerRect[2], self._innerRect[3]))

        self.__setExtremes(x, y, self._viewPortLimit[0], self._viewPortLimit[1], True)
        return self

    def addOffset(self, offset_x=0, offset_y=0, vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        xmin, xmax = (self._innerRect[0], self._innerRect[2]) if use_vp_fractions else (self.xmin(), self.xmax())
        ymin, ymax = (self._innerRect[1], self._innerRect[3]) if use_vp_fractions else (self.ymin(), self.ymax())
        self.__setExtremes((xmin + offset_x, xmax + offset_x), (ymin + offset_y, ymax + offset_y), vp_xmax, vp_ymax,
                           use_vp_fractions)
        return self

    def addScaleFactor(self, factor_x=1, factor_y=1, vp_xmax=None, vp_ymax=None, use_vp_fractions=False):
        xmin, xmax = (self._innerRect[0], self._innerRect[2]) if use_vp_fractions else (self.xmin(), self.xmax())
        ymin, ymax = (self._innerRect[1], self._innerRect[3]) if use_vp_fractions else (self.ymin(), self.ymax())
        self.__setExtremes((xmin * factor_x, xmax * factor_x), (ymin * factor_y, ymax * factor_y), vp_xmax, vp_ymax,
                           use_vp_fractions)
        return self

    def addBorder(self, borderx=0, bordery=0, use_vp_fractions=False, expand=True):
        _borderx, _bordery = (
            borderx / self._viewPortLimit[0], bordery / self._viewPortLimit[1]) if not use_vp_fractions else (
            borderx, bordery)
        borderx, bordery = (
            self._viewPortLimit[0] * _borderx, self._viewPortLimit[1] * _bordery) if use_vp_fractions else (
            borderx, bordery)

        if self.xmin() < borderx and expand:
            self._innerRect = (0, self._innerRect[1], self._innerRect[2], self._innerRect[3])
            self.changeViewportLimits(self._viewPortLimit[0] + borderx - self.xmin(), self._viewPortLimit[1])
        else:
            self._innerRect = (
                max(0, self._innerRect[0] - _borderx), self._innerRect[1], self._innerRect[2], self._innerRect[3])

        if borderx > (self._viewPortLimit[0] - self.xmax()) and expand:
            self.changeViewportLimits(borderx + self.xmax(), self._viewPortLimit[1])
            self._innerRect = (self._innerRect[0], self._innerRect[1], 1, self._innerRect[3])
        else:
            self._innerRect = (
                self._innerRect[0], self._innerRect[1], min(1, self._innerRect[2] + _borderx), self._innerRect[3])

        if self.ymin() < bordery and expand:
            self._innerRect = (self._innerRect[0], 0, self._innerRect[2], self._innerRect[3])
            self.changeViewportLimits(self._viewPortLimit[0], self._viewPortLimit[1] + bordery - self.ymin())
        else:
            self._innerRect = (
                self._innerRect[0], max(0, self._innerRect[1] - _bordery), self._innerRect[2], self._innerRect[3])

        if bordery > self._viewPortLimit[1] - self.ymax() and expand:
            self.changeViewportLimits(self._viewPortLimit[0], bordery + self.ymax())
            self._innerRect = (self._innerRect[0], self._innerRect[1], self._innerRect[2], 1)
        else:
            self._innerRect = (
                self._innerRect[0], self._innerRect[1], self._innerRect[2], min(1, self._innerRect[3] + _bordery))

    def toBox(self, order='xy', sequence='minmax', use_vp_fractions=False):
        vp_xmax, vp_ymax = (1, 1) if use_vp_fractions else self._viewPortLimit
        xmin, ymin, xmax, ymax = self.xmin(vp_xmax), self.ymin(vp_ymax), self.xmax(vp_xmax), self.ymax(vp_ymax)

        if order == "xy":
            if sequence == "minmax":
                return xmin, ymin, xmax, ymax
            elif sequence == "maxmin":
                return xmax, ymax, xmin, ymin
            else:
                raise ValueError("Invalid coordinate sequence requested")
        elif order == 'yx':
            if sequence == "minmax":
                return ymin, xmin, ymax, xmax
            elif sequence == "maxmin":
                return ymax, xmax, ymin, xmin
            else:
                raise ValueError("Invalid coordinate sequence requested")
        elif order == 'xx':
            if sequence == "minmax":
                return xmin, xmax, ymin, ymax
            elif sequence == "maxmin":
                return xmax, xmin, ymax, ymin
            else:
                raise ValueError("Invalid coordinate sequence requested")
        elif order == 'yy':
            if sequence == "minmax":
                return ymin, ymax, xmin, xmax
            elif sequence == "maxmin":
                return ymax, ymin, xmax, xmin
            else:
                raise ValueError("Invalid coordinate sequence requested")
        else:
            raise ValueError("Invalid coordinate order requested")

    def to2Points(self, order='xy', sequence='minmax', use_vp_fractions=False):
        if not order in ('xy', 'yx'): order = 'xy'
        c1, c2, c3, c4 = self.toBox(order, sequence, use_vp_fractions)
        return (c1, c2), (c3, c4)

    def to4Points(self, order='xy', sequence='minmax', use_vp_fractions=False):
        if not order in ('xy', 'yx'): order = 'xy'
        c1, c2, c3, c4 = self.toBox(order, sequence, use_vp_fractions)
        return (c1, c2), (c3, c4), (c1, c4), (c3, c2)

    def center(self, order='xy'):
        return self.__pointProvider((self.xmin() + self.xmax()) / 2, (self.ymin() + self.ymax()) / 2, order)

    def topLeft(self, order='xy'):
        return self.__pointProvider(self.xmin(), self.ymin(), order)

    def bottomRight(self, order='xy'):
        return self.__pointProvider(self.xmax(), self.ymax(), order)

    def __pointProvider(self, x, y, order):
        return (int(x), int(y)) if order == 'xy' else (int(y), int(x))

    def union(self, rect=None, std_box=None):
        x = [self.xmin(), self.xmax()]
        y = [self.ymin(), self.ymax()]

        if not rect is None:
            new_viewport = max(self._viewPortLimit[0], rect._viewPortLimit[0]), max(self._viewPortLimit[1],
                                                                                    rect._viewPortLimit[1])
            self.changeViewportLimits(new_viewport[0], new_viewport[1])
            rect.changeViewportLimits(new_viewport[0], new_viewport[1])

            x.extend([rect.xmin(), rect.xmax()])
            y.extend([rect.ymin(), rect.ymax()])
        elif not std_box is None:
            x.extend([std_box[0], std_box[2]])
            y.extend([std_box[1], std_box[3]])

        self.__setExtremes(x, y)
        return self

    def getDistFromRect(self, rect, reference="border", type="cartesian"):
        return self._getDistFromStdBox(box=rect.toBox(), reference=reference, type=type)

    def _getDistFromStdBox(self, box, reference="border", type="cartesian"):
        return _stdBoxesOps().stdBoxesDistance(self.toBox(), box, reference=reference, type=type)


class _stdBoxesOps():
    def stdBoxesDistance(self, box1, box2, reference="border", type="cartesian"):
        if reference == 'center':
            x0, y0 = box1.center()
            x1, y1 = box2.center()
        elif reference == 'border':
            x0, y0 = 0, 0
            x1 = box2[0] - box1[2] if box1[2] < box2[0] else box1[0] - box2[2] if box1[0] > box2[2] else 0
            y1 = box2[1] - box1[3] if box1[3] < box2[1] else box1[1] - box2[3] if box1[1] > box2[3] else 0
        else:
            raise ValueError("Invalid reference to calculate distance")
        return _stdPointsOps().pointsDistance(x0, y0, x1, y1, type=type)

    def stdBoxesEqual(self, box1, box2):
        return box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]

    def isStdBoxOverlapped(self, box1, box2, tolerance=5):
        def is_inside(r1, r2):
            return r1[0] + tolerance >= r2[0] and r1[1] + tolerance >= r2[1] and r1[2] <= r2[2] + tolerance and r1[3] <= \
                r2[3] + tolerance

        return is_inside(box1, box2) or is_inside(box2, box1)


class _stdPointsOps():
    def pointsDistance(self, x0=0, y0=0, x1=0, y1=0, type="cartesian"):
        if type == "cartesian":
            return int(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        elif type == "cartesian_squares":
            return (x1 - x0) ** 2 + (y1 - y0) ** 2
        elif type == "manhattan":
            return abs(x1 - x0) + abs(y1 - y0)
        else:
            raise ValueError("Unknown distance type specified")


class Rects:
    """
    A class to manage a collection of Rect objects with various utility methods for manipulation and calculation.

    Attributes:
    ----------
    viewPort : Rect
        The viewport rectangle.
    tempRect : Rect
        A temporary rectangle for internal use.
    inboundMapping : dict
        Mapping of inbound methods.
    outboundMapping : dict
        Mapping of outbound methods.
    rects : list
        List of rectangles.

    Methods:
    -------
    __init__(self):
        Initializes the Rects object.
    __getattr__(self, method):
        Dynamically handles method calls for inbound and outbound mappings.
    __str__(self):
        Returns a string representation of the Rects object.
    __len__(self):
        Returns the number of rectangles.
    updateViewport(self):
        Updates the viewport rectangle based on the current rectangles.
    addRect(self, rect, update_viewport=True):
        Adds a rectangle to the collection.
    simplifyRects(self, tolerance=5):
        Simplifies the collection of rectangles by merging close ones.
    moveRectsFrom(self, rects):
        Moves rectangles from another Rects object.
    _addStdBox(self, box, update_viewport=True):
        Adds a standard box to the collection.
    _moveStdBoxesFrom(self, boxes):
        Moves standard boxes from another collection.
    __getRect(self, index, then_remove=False):
        Retrieves a rectangle by index.
    getRect(self, index):
        Retrieves a rectangle by index without removing it.
    popRect(self, index):
        Retrieves and removes a rectangle by index.
    """

    def __init__(self, id=None):
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
        self.id = id if not id is None else None

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
        return (f"SRects ({len(self.rects)} Rects, overall top_left:[{self.viewPort.xmin()}, {self.viewPort.ymin()}] "
                f"overall bottom_right:[{self.viewPort.xmax()}, {self.viewPort.ymax()}] "
                f"overall size:[{self.viewPort.width()}, {self.viewPort.height()}] "
                f"viewport:[{self.viewPort.getViewPortLimit()}])")

    def __len__(self):
        return len(self.rects)

    def updateViewport(self, vp_xmax=0, vp_ymax=0):
        if len(self.rects):
            self.viewPort = Rect(box=(min([rect[0] for rect in self.rects]), min([rect[1] for rect in self.rects]),
                                      max([rect[2] for rect in self.rects]), max([rect[3] for rect in self.rects])))
        if vp_xmax or vp_ymax:
            self.viewPort.union(std_box=(0, 0, vp_xmax, vp_ymax))

    def addRect(self, rect, update_viewport=True):
        return self._addStdBox(box=rect.toBox(), update_viewport=update_viewport)

    def _addStdBox(self, box, update_viewport=True):
        self.rects.append(box)
        if update_viewport:
            if self.viewPort is None:
                self.viewPort = Rect(box=box)
            self.viewPort.union(std_box=box)
        return True

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
            while reset:
                reset = False
                for j in range(len(self.rects) - 1, -1, -1):
                    this_rect = self.rects.pop(j)
                    if not can_merge((1, 3), (0, 2)) and not can_merge((0, 2), (1, 3)):
                        rect_new.append(this_rect)
                    else:
                        rect_new[-1] = min(this_rect[0], rect_new[-1][0]), min(this_rect[1], rect_new[-1][1]), max(
                            this_rect[2], rect_new[-1][2]), max(this_rect[3], rect_new[-1][3])
                        reset = True
                        break
        self.rects = rect_new
        self.updateViewport()
        return self

    def moveRectsFrom(self, rects):
        res = self._moveStdBoxesFrom(boxes=rects.rects, maybe_update_viewport=False)
        if res:
            self.viewPort.union(std_box=rects.viewPort.toBox())
        return res

    def removeOverlappingRects(self, rects):
        res=[]
        for i in range(len(self.rects) - 1, -1, -1):
            for rect in rects.rects:
                if _stdBoxesOps().isStdBoxOverlapped(self.rects[i], rect):
                    self.rects.pop(i)
                    res.append(i)
                    break
        self.updateViewport()
        return res

    def _moveStdBoxesFrom(self, boxes, maybe_update_viewport=True):
        startLen = len(boxes)
        for i in range(len(boxes) - 1, -1, -1):
            if self._addStdBox(boxes[i], update_viewport=False):
                boxes.pop(i)
        if startLen > len(boxes):
            if maybe_update_viewport:
                self.updateViewport()
            return True
        return False

    def __getRect(self, index, then_remove=False):
        rect = Rect().fromBox(self.rects[index])
        if then_remove:
            del self.rects[index]
            self.updateViewport()
        return rect

    def getRect(self, index):
        return self.__getRect(index, then_remove=False)

    def popRect(self, index):
        return self.__getRect(index, then_remove=True)


class CactusRects(Rects):
    """
    A specialized class derived from Rects to handle specific operations with tolerance and strategy.

    Attributes:
    ----------
    tolerance : int
        The tolerance value for operations.
    strategy : str
        The strategy for handling rectangles.

    Methods:
    -------
    __init__(self, seedRect, tolerance=5, strategy="full"):
        Initializes the CactusRects object with a seed rectangle, tolerance, and strategy.
    addRect(self, rect, update_viewport=True):
        Adds a rectangle to the collection with specific tolerance and strategy.
    _addStdBox(self, box, update_viewport=True):
        Adds a standard box to the collection with specific tolerance and strategy.
    moveRectsFrom(self, rects):
        Moves rectangles from another Rects object.
    _moveStdBoxesFrom(self, boxes):
        Moves standard boxes from another collection.
    __getattr__(self, method):
        Dynamically handles method calls for inbound mappings.
    """

    def __init__(self, seedRect, tolerance=5, strategy="full", id=None):
        super().__init__(id=id)
        super()._addStdBox(seedRect.toBox())
        self.tolerance = tolerance
        self.square_tolerance = self.tolerance ** 2
        self.strategy = strategy
        self.stopper_callback = None

    def setStopperCallback(self, callback):
        self.stopper_callback = callback

    def addRect(self, rect, update_viewport=True):
        if not self._stdBoxIsCloseToVP(rect.toBox()):
            return False
        return self._maybeAddStdBoxToSearchSet(box=rect.toBox(), update_viewport=update_viewport)

    def _addStdBox(self, box, update_viewport=True):
        if not self._stdBoxIsCloseToVP(box):
            return False
        return self._maybeAddStdBoxToSearchSet(box=box, update_viewport=update_viewport)

    def _maybeAddStdBoxToSearchSet(self, box, update_viewport=True):
        src_set = self._getSearchSet()
        for boundary_rect in src_set:
            rect_dist = _stdBoxesOps().stdBoxesDistance(box, boundary_rect, reference="border",
                                                        type="cartesian_squares")
            if rect_dist <= self.square_tolerance:
                super()._addStdBox(box, update_viewport=update_viewport)
                return True
        return False

    def _stdBoxIsCloseToVP(self, box):
        return not self.viewPort._getDistFromStdBox(box, reference="border",
                                                    type="cartesian_squares") > self.square_tolerance

    def _getSearchSet(self):
        boundary_rects = set()
        xmin, ymin, xmax, ymax = self.viewPort.xmin(), self.viewPort.ymin(), self.viewPort.xmax(), self.viewPort.ymax()
        if self.strategy == 'full':
            boundary_rects.update(r for r in self.rects)
        elif self.strategy == 'boundaries_only':
            boundary_rects.update(r for r in self.rects if r[0] == xmin or r[1] == ymin or r[2] == xmax or r[3] == ymax)
        else:
            raise ValueError("Unknown merge strategy")

        return boundary_rects

    def moveRectsFrom(self, rects):
        while super().moveRectsFrom(rects):
            if not self.stopper_callback is None:
                if self.stopper_callback(self.id, self.rects[-1]) == True:
                    break

    def _moveStdBoxesFrom(self, boxes):
        while super()._moveStdBoxesFrom(boxes):
            if not self.stopper_callback is None:
                if self.stopper_callback(self.id, self.rects[-1]) == True:
                    break

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
