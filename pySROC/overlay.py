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


class Rect:
    def __init__(self, points=None, box=None, order='xy', xref=None, yref=None, fractions=False):
        if box:
            self.fromBox(box=box, order=order, xref=xref, yref=yref, fractions=fractions)
        elif points:
            self.fromPoints(points=points, order=order, xref=xref, yref=yref, fractions=fractions)
        else:
            self.fromBox((0, 0, 0, 0,))

    def __str__(self):
        return f"SRect (top_left:[{self.xmin()}, {self.ymin()}] bottom_right:[{self.xmax()}, {self.ymax()}] size:[{self.width()}, {self.height()}] viewport:[{self.xref}, {self.yref}])"

    def __setExtremes(self, x, y, xref=None, yref=None, fractions=False):
        xmax = max(x)
        ymax = max(y)
        if not xref:
            if xmax:
                xref = xmax
            else:
                xref = 1

        if not yref:
            if ymax:
                yref = ymax
            else:
                yref = 1

        self.xref = xref
        self.yref = yref
        if fractions:
            xref = 1
            yref = 1

        self.__xmin = min(x) / xref
        self.__ymin = min(y) / yref
        self.__xmax = xmax / xref
        self.__ymax = ymax / yref

    def __validateOrder(self, order):
        if order not in ['xy', 'yx', 'xx', 'yy']:
            raise ValueError(f"Invalid coordinate order requested")

    def __validateSequence(self, sequence):
        if sequence not in ['minmax', 'maxmin']:
            raise ValueError(f"Invalid coordinate sequence requested")

    def __validateRect(self, rect):
        if not isinstance(rect, type(self)):
            raise ValueError(f"Rectangle is not a {type(self)} instance")

    def scaleReference(self, xref, yref):
        self.__xmin = min(1, self.__xmin * self.xref / xref)
        self.__xmax = min(1, self.__xmax * self.xref / xref)
        self.__ymin = min(1, self.__ymin * self.yref / yref)
        self.__ymax = min(1, self.__ymax * self.yref / yref)

        self.xref = xref
        self.yref = yref

    def setReference(self, xref, yref):
        self.xref = xref
        self.yref = yref

    def fromBox(self, box=(0, 0, 0, 0), order='xy', xref=None, yref=None, fractions=False):
        self.__validateOrder(order)
        if order == 'xy':
            x = (box[0], box[2])
            y = (box[1], box[3])
        elif order == 'yx':
            x = (box[1], box[3])
            y = (box[0], box[2])
        elif order == 'xx':
            x = (box[0], box[1])
            y = (box[2], box[3])
        elif order == 'yy':
            x = (box[2], box[3])
            y = (box[0], box[1])
        self.__setExtremes(x, y, xref, yref, fractions)
        return self

    def fromPoints(self, points=((0, 0), (0, 0), (0, 0), (0, 0)), order='xy', xref=None, yref=None, fractions=False):
        # TODO: restrict more valid orders for points
        self.__validateOrder(order)
        index_x = order == 'yx'

        x = tuple(point[index_x] for point in points)
        y = tuple(point[not index_x] for point in points)

        self.__setExtremes(x, y, xref, yref, fractions)
        return self

    def width(self):
        return abs(self.xmax() - self.xmin())

    def height(self):
        return abs(self.ymax() - self.ymin())

    def widthRef(self):
        return self.xref

    def heightRef(self):
        return self.yref

    def area(self):
        return self.width() * self.height()

    def perimeter(self):
        return 2 * (self.width() + self.height())

    def rotate(self, rotation, xr=0, yr=0, fractions=False):
        if rotation % 90:
            raise ValueError(f"Multiple of 90° are only accepted for straight rectangles")

        rad = rotation * math.pi / 180
        if not fractions:
            xr /= self.xref
            yr /= self.yref

        def rotate_x(x, y):
            return (x - xr) * math.cos(rad) - (y - yr) * math.sin(rad) + xr

        def rotate_y(x, y):
            return (x - xr) * math.sin(rad) + (y - yr) * math.cos(rad) + yr

        x = (rotate_x(self.__xmin, self.__ymin), rotate_x(self.__xmax, self.__ymax))
        y = (rotate_y(self.__xmin, self.__ymin), rotate_y(self.__xmax, self.__ymax))

        # if min(x) < 0 or min(y) < 0:
        #    raise ValueError(f"Requested rotation gives negative coordinates")
        # TODO check if need to rotate also xref yref
        self.__setExtremes(x, y, self.xref, self.yref, True)
        return self

    def addOffset(self, offset_x=0, offset_y=0, xref=None, yref=None, fractions=False):
        if fractions:
            xmin = self.__xmin
            xmax = self.__xmax
            ymin = self.__ymin
            ymax = self.__ymax
        else:
            xmin = self.xmin()
            xmax = self.xmax()
            ymin = self.ymin()
            ymax = self.ymax()

        self.__setExtremes((xmin + offset_x, xmax + offset_x),
                           (ymin + offset_y, ymax + offset_y),
                           xref, yref, fractions)
        return self

    def addScaleFactor(self, factor_x=1, factor_y=1, xref=None, yref=None, fractions=False):
        if fractions:
            xmin = self.__xmin
            xmax = self.__xmax
            ymin = self.__ymin
            ymax = self.__ymax
        else:
            xmin = self.xmin()
            xmax = self.xmax()
            ymin = self.ymin()
            ymax = self.ymax()

        self.__setExtremes((xmin * factor_x, xmax * factor_x),
                           (ymin * factor_y, ymax * factor_y),
                           xref, yref, fractions)
        return self

    def addBorder(self, borderx=0, bordery=0, fractions=False, expand=True):
        if not fractions:
            __borderx = borderx / self.xref
            __bordery = bordery / self.yref
        else:
            __borderx = borderx
            __bordery = bordery
            borderx = self.xref * __borderx
            bordery = self.yref * __bordery

        if self.xmin() < borderx and expand:
            self.__xmin = 0
            self.scaleReference(self.xref + borderx - self.xmin(), self.yref)
        else:
            self.__xmin = max(0, self.__xmin - __borderx)

        if borderx > (self.xref - self.xmax()) and expand:
            self.scaleReference(borderx + self.xmax(), self.yref)
            self.__xmax = 1
        else:
            self.__xmax = min(1, self.__xmax + __borderx)

        if self.ymin() < bordery and expand:
            self.__ymin = 0
            self.scaleReference(self.xref, self.yref + bordery - self.ymin())
        else:
            self.__ymin = max(0, self.__ymin - __bordery)

        if bordery > self.yref - self.ymax() and expand:
            self.scaleReference(self.xref, bordery + self.ymax())
            self.__ymax = 1
        else:
            self.__ymax = min(1, self.__ymax + __bordery)

    # EXPORTS
    def toBox(self, order='xy', sequence='minmax', fractions=False):
        self.__validateOrder(order)
        self.__validateSequence(sequence)
        if fractions:
            xref = 1
            yref = 1
        else:
            xref = self.xref
            yref = self.yref

        if order == "xy":
            if sequence == "minmax":
                return self.xmin(), self.ymin(), self.xmax(), self.ymax()
            else:
                return self.xmax(), self.ymax(), self.xmin(), self.ymin()
        elif order == 'yx':
            if sequence == "minmax":
                return self.ymin(), self.xmin(), self.ymax(), self.xmax()
            else:
                return self.ymax(), self.xmax(), self.ymin(), self.xmin()
        elif order == 'xx':
            if sequence == "minmax":
                return self.xmin(), self.xmax(), self.ymin(), self.ymax()
            else:
                return self.xmax(), self.xmin(), self.ymax(), self.ymin()
        elif order == 'yy':
            if sequence == "minmax":
                return self.ymin(), self.ymax(), self.xmin(), self.xmax()
            else:
                return self.ymax(), self.ymin(), self.xmax(), self.xmin()

    def to2Points(self, order='xy', sequence='minmax', fractions=False):
        c1, c2, c3, c4 = self.toBox(order, sequence, fractions)
        return (c1, c2), (c3, c4)

    def to4Points(self, order='xy', sequence='minmax', fractions=False):
        c1, c2, c3, c4 = self.toBox(order, sequence, fractions)
        return (c1, c2), (c3, c4), (c1, c4), (c3, c2)

    # SINGLE POINTS
    def xmin(self, xref=None):
        if xref is None:
            xref = self.xref
        return int(self.__xmin * xref)

    def xmax(self, xref=None):
        if xref is None:
            xref = self.xref
        return int(self.__xmax * xref)

    def ymin(self, yref=None):
        if yref is None:
            yref = self.yref
        return int(self.__ymin * yref)

    def ymax(self, yref=None):
        if yref is None:
            yref = self.yref
        return int(self.__ymax * yref)

    def center(self, order='xy'):
        self.__pointProvider((self.xmin() + self.xmax()) / 2, (self.ymin() + self.ymax()) / 2)

    def topLeft(self, order='xy'):
        return self.__pointProvider(self.xmin(), self.ymin(), order)

    def bottomRight(self, order='xy'):
        return self.__pointProvider(self.xmax(), self.ymax(), order)

    def __pointProvider(self, x, y, order):
        self.__validateOrder(order)
        # TODO: restrict more valid orders for points
        if order == 'xy':
            return int(x), int(y)
        else:
            return int(y), int(x)

    def __pointsDistance(self, x0=0, y0=0, x1=0, y1=0):
        return int(math.sqrt(abs(x1 - x0) ** 2 + abs(y1 - y0) ** 2))

    # OPERATIONS WITH OTHER RECTS
    def union(self, rect):
        self.__validateRect(rect)

        x = (min(self.xmin(), rect.xmin()), max(self.xmax(), rect.xmax()))
        y = (min(self.ymin(), rect.ymin()), max(self.ymax(), rect.ymax()))
        xref = max(self.xref, rect.xref)
        yref = max(self.yref, rect.yref)

        self.__setExtremes(x=x, y=y, xref=xref, yref=yref)
        return self

    def getDistFromRect(self, rect, reference="border"):
        self.__validateRect(rect)
        if reference == 'center':
            x0, y0 = rect.center()
            x1, y1 = self.center()
        elif reference == 'border':
            x0 = 0
            y0 = 0
            if self.xmax() < rect.xmin():
                x1 = rect.xmin() - self.xmax()
            elif self.xmin() > rect.xmax():
                x1 = self.xmin() - rect.xmax()
            else:
                x1 = 0
            if self.ymax() < rect.ymin():
                y1 = rect.ymin() - self.ymax()
            elif self.ymin() > rect.ymax():
                y1 = self.ymin() - rect.ymax()
            else:
                y1 = 0
        else:
            raise ValueError(f"Invalid reference to calculate distance")

        return self.__pointsDistance(x0, y0, x1, y1)


class Rects:
    def __init__(self):
        self.mainRect = None
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

    def __getattr__(self, method):
        if method in self.inboundMapping:
            inboundMethod = getattr(self.tempRect, self.inboundMapping[method])

            def wrapper(*args, **kwargs):
                if args:
                    rects = args[0]
                    args = args[1:]
                    for rect in rects:
                        self.addRect(inboundMethod(rect, *args, **kwargs))
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
        return f"SRects ({len(self.rects)} Rects, overall top_left:[{self.mainRect.xmin()}, {self.mainRect.ymin()}] overall bottom_right:[{self.mainRect.xmax()}, {self.mainRect.ymax()}] overall size:[{self.mainRect.width()}, {self.mainRect.height()}] viewport:[{self.mainRect.xref}, {self.mainRect.yref}])"

    def __len__(self):
        return len(self.rects)

    def updateViewport(self):
        if len(self.rects):
            self.mainRect = Rect().fromBox(self.rects[0])
        for rect in self.rects:
            self.mainRect.union(Rect().fromBox(rect))

    def validateRect(self, rect):
        self.tempRect._Rect__validateRect(rect)

    def addRect(self, rect):
        self.validateRect(rect)
        self.rects.append((rect.toBox()))
        if self.mainRect is None:
            self.mainRect = Rect().fromBox(rect.toBox())
        else:
            self.mainRect.union(rect)
        # self.updateViewport()
        return True

    def moveRectsFrom(self, rects):
        startLen = len(rects)
        i = 0
        while i < len(rects):
            if self.addRect(Rect().fromBox(rects.rects[i])):
                rects.rects.pop(i)
            else:
                i += 1
        return startLen > len(rects)

    def __getRect(self, index, then_remove=False):
        # if index in range(len(self)):
        rect = Rect().fromBox(self.rects[index])
        if then_remove:
            del (self.rects[index])
        return rect

    def getRect(self, index):
        return self.__getRect(index, then_remove=False)

    def popRect(self, index):
        return self.__getRect(index, then_remove=True)


class CactusRects(Rects):
    def __init__(self, seedRect, tolerance=5):
        super().__init__()
        super().addRect(seedRect)
        # self.seed = seedRect
        self.tolerance = tolerance

    def addRect(self, rect):
        for branch in self.rects:
            if rect.getDistFromRect(self.tempRect.fromBox(branch), reference="border") <= self.tolerance:
                super().addRect(rect)
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
