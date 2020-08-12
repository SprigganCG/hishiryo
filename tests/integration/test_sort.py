"""Integration tests for the sort option."""
import pytest
from hishiryo import Hishiryo
import os


# simple run against a test csv dataset and sort by
# different criteria
def test_render_dot_bitmap_from_csv():

    # define test case
    input_path = os.path.dirname(os.path.realpath(__file__)) \
        + "/../fixtures/titanic-dataset-train.csv"
    output_path = os.path.dirname(
        os.path.realpath(__file__)) + "/output-sort.png"
    separator = ","
    radius = 3000
    sort_by = ["Survived", "Pclass", "Sex", "Age"]
    glyph_type = "Polygon"
    print(os.path.realpath(__file__),
          os.path.dirname(os.path.realpath(__file__)))

    # Instanciate Hishiryo
    HishiryoConverter = Hishiryo.Hishiryo()
    assert HishiryoConverter.convertCSVToRadialBitmap(input_path,
                                                      separator,
                                                      output_path,
                                                      radius,
                                                      sort_by,
                                                      glyph_type
                                                      )
