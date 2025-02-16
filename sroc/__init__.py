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

#import logging

from .overlay import Rect
from .overlay import Rects
from .overlay import CactusRects
from .canvas import RectCanvas
from .canvas import RectScanner
from .canvas import CanvasTiler

__version__ = '0.1.0'
__author__ = 'Giovanni Cascione'

# Configura il logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.info('Il pacchetto è stato inizializzato')

__all__ = [
    'Rect',
    'Rects',
    'CactusRects',
    'RectCanvas',
    'RectScanner',
    'CanvasTiler'
]
