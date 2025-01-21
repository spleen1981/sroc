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

import os

def test_saved_file(file_path,silent_success=False):
    if os.path.exists(file_path):
        if not silent_success:
            print(f"File '{file_path}' created successfully.")
        return True
    else:
        print(f"Error: file '{file_path}' not created.")
        return False

def get_path_info(filename_path):
    path = os.path.dirname(filename_path)
    path.replace('\\', '/')
    split_path, ext = os.path.splitext(filename_path)

    name = os.path.basename(split_path)
    return path, name, ext