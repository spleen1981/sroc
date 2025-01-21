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


class typeList(list):
    def __init__(self, type):
        self.type = type
        super().__init__()

    def append(self, item):
        if not isinstance(item, self.type):
            raise TypeError(
                f"Item of type {type(item).__name__} not allowed. Only {self.type.__name__} are allowed.")
        super().append(item)

    def extend(self, items):
        for item in items:
            if not isinstance(item, self.type):
                raise TypeError(
                    f"Item of type {type(item).__name__} not allowed. Only {self.type.__name__} are allowed.")
        super().extend(items)


class Table():
    def __init__(self, dictionary=None):
        self.data = {}
        if dictionary:
            self.data = dictionary

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        if not self.getKeys():
            return 0
        else:
            return max([len(self.data[key]) for key in self.getKeys()])

    def __add__(self, table):
        res = {}
        for key in self.data:
            if key in table.data:
                res[key] = self.data[key] + table.data[key]
            else:
                res[key] = self.data[key]
        for key in table.data:
            if key not in res:
                res[key] = table.data[key]
        return self.__class__(dictionary=res)

    def initDefaultKey(self, key, value, dictionary=None):
        if value is None and dictionary is not None and dictionary.get(key):
            return
        elif value is None:
            value = []

        self.addColumn(key, value)

    def addColumn(self, key, data=None):
        if data is None:
            data = []
        if not len(self) or not len(data) > len(self):
            self.data.update({key: data})

    def getKeys(self):
        return [key for key, _ in self.data.items()]

    def extractRowByIndex(self, index):
        return [self.data[key][index] for key in self.getKeys()]

    def removeRowByIndex(self, index):
        for key in self.getKeys():
            del self.data[key][index]

    def extractAllRows(self):
        return [self.extractRowByIndex(index) for index in range(len(self))]

    def renameKey(self, old_key, new_key):
        self.data[new_key] = self.data.pop(old_key)

    def toDict(self):
        return self.data

    def fromDict(self, dictionary):
        self.__init__(dictionary)
        return self

    def removeRowLowerThan(self, key, value):
        for i in range(len(self.data[key]) - 1, -1, -1):
            if self.data[key][i] < value:
                self.removeRowByIndex(i)


class tilesCrawlerResult(Table):
    def __init__(self, dictionary=None, offsetBoxes=None):
        super().__init__(dictionary)
        self.initDefaultKey('offsetBoxes', offsetBoxes, dictionary)
