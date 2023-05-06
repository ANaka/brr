import os
import zipfile
from pathlib import Path

import numpy as np
import requests
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from geodude import fonts_root
from geodude.line import bezier_func
from shapely import Polygon
from shapely.geometry import Polygon


def get_font_paths():
    """
    Returns a dictionary of font names and paths.
    """
    return {f.stem: f for f in fonts_root.glob("**/*.ttf")}


FONT_PATHS = get_font_paths()


def get_font_path(font_name):
    """
    Returns the path to the specified font.

    Args:
        font_name (str): The name of the font.

    Returns:
        str: The path to the font.
    """
    return FONT_PATHS[font_name]


def download_font(font_name, target_directory: str = None):
    if target_directory is None:
        target_directory = fonts_root / font_name.replace(" ", "_")

    target_directory = Path(target_directory)
    url = f"https://fonts.google.com/download?family={font_name.replace(' ', '+')}"
    response = requests.get(url)

    # ensure the request was successful
    response.raise_for_status()

    # ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # write the content of the response to a file
    filename = f"{font_name.replace(' ', '_')}.zip"
    filepath = target_directory / filename
    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"{font_name} downloaded successfully to {filepath}!")

    # unzip the file
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(target_directory)

    print(f"{font_name} unzipped successfully in {target_directory}!")


def get_glyph(font_name, glyph_name):
    font_path = get_font_path(font_name)
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    return glyph_set[glyph_name]


def extract_glyph_points(font_name, glyph_name, n_points=100):
    """
    Extracts the glyph points from the specified font file.

    Args:
        font_path (str): Path to the font file.
        glyph_name (str): Name of the glyph to extract.
        n_points (int): Number of points to sample from each curve.

    Returns:
        list: A list of subpaths, where each subpath is a list of (x, y) tuples representing the points of the glyph.
    """
    # Load the font and get the glyph
    glyph = get_glyph(font_name, glyph_name)

    # Draw the glyph with a RecordingPen
    pen = RecordingPen()
    glyph.draw(pen)

    # Get the commands from the pen
    commands = pen.value

    # Create a list to store the subpaths
    subpaths = []
    # And a list to store the points of the current subpath
    subpath = []

    # For each command
    for command in commands:
        operation, command_points = command
        # If the command is a curve, use the Bezier function to get the points
        if operation in ["curveTo", "qCurveTo"]:
            bezier = bezier_func(command_points)
            t_values = np.linspace(0, 1, n_points)
            curve_points = bezier(t_values, as_numpy=True)
            subpath.extend(curve_points)
        # If the command is a line or move, just use the command points
        elif operation in ["lineTo", "moveTo"]:
            subpath.extend(command_points)
        # If the command is a closePath, finish the current subpath and start a new one
        elif operation == "closePath":
            if subpath:
                subpaths.append(subpath)
                subpath = []

    # If there's an open subpath at the end, add it to the list
    if subpath:
        subpaths.append(subpath)

    return subpaths


def get_character_polygon(font_name, glyph_name, n_points=100):
    """
    Returns a shapely Polygon representing the glyph.

    Args:
        font_path (str): Path to the font file.
        glyph_name (str): Name of the glyph to extract.
        n_points (int): Number of points to sample from each curve.

    Returns:
        shapely.geometry.Polygon: A polygon representing the glyph.
    """
    # Get the glyph points
    subpaths = extract_glyph_points(font_name, glyph_name, n_points=n_points)
    # Create a polygon from the points
    polygon = Polygon(subpaths[0])

    if not polygon.is_valid:
        polygon = polygon.buffer(1e-19)

    # If there are any holes, add them to the polygon
    for subpath in subpaths[1:]:
        subpath_polygon = Polygon(subpath)
        if subpath_polygon.is_valid:
            polygon = polygon.difference(subpath_polygon)
        else:
            # If the subpath isn't valid, try to fix it with a buffer
            subpath_polygon = subpath_polygon.buffer(1e-19)

        polygon = polygon.difference(subpath_polygon)
    return polygon
