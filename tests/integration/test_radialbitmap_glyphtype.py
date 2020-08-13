"""Integration tests for the glyph option."""
import pytest
from hishiryo import Hishiryo
import os


# simple run against a test csv dataset to generate dot bitmap
def test_render_dot_bitmap_from_csv():

    # define test case
    input_path = os.path.dirname(os.path.realpath(
        __file__)) + "/../fixtures/titanic-dataset-train.csv"
    output_path = os.path.dirname(
        os.path.realpath(__file__)) + "/output-dot.png"
    separator = ","
    radius = 3000
    sort_by = "PassengerId"
    glyph_type = "Dot"

    # Instanciate Hishiryo
    HishiryoConverter = Hishiryo.Hishiryo()
    assert HishiryoConverter.convertCSVToRadialBitmap(input_path,
                                                      separator,
                                                      output_path,
                                                      radius,
                                                      sort_by,
                                                      glyph_type
                                                      )


# simple run against a test csv dataset to generate square bitmap
def test_render_square_bitmap_from_csv():

    # define test case
    input_path = os.path.dirname(os.path.realpath(
        __file__)) + "/../fixtures/titanic-dataset-train.csv"
    output_path = os.path.dirname(
        os.path.realpath(__file__)) + "/output-square.png"
    separator = ","
    radius = 3000
    sort_by = "PassengerId"
    glyph_type = "Square"

    # Instanciate Hishiryo
    HishiryoConverter = Hishiryo.Hishiryo()
    assert HishiryoConverter.convertCSVToRadialBitmap(input_path,
                                                      separator,
                                                      output_path,
                                                      radius,
                                                      sort_by,
                                                      glyph_type
                                                      )


# simple run against a test csv dataset to generate polygon bitmap
def test_render_polygon_bitmap_from_csv():

    # define test case
    input_path = os.path.dirname(os.path.realpath(
        __file__)) + "/../fixtures/titanic-dataset-train.csv"
    output_path = os.path.dirname(
        os.path.realpath(__file__)) + "/output-polygon.png"
    separator = ","
    radius = 3000
    sort_by = "PassengerId"
    glyph_type = "Polygon"

    # Instanciate Hishiryo
    HishiryoConverter = Hishiryo.Hishiryo()
    assert HishiryoConverter.convertCSVToRadialBitmap(input_path,
                                                      separator,
                                                      output_path,
                                                      radius,
                                                      sort_by,
                                                      glyph_type
                                                      )
