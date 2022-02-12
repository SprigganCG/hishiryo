"""Integration tests for the sort option."""
import pytest
from hishiryo import Hishiryo
import os


def test_render_bitmap_from_csv():
    """Render a simple square bitmap from a csv"""

    # define test case
    input_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/../fixtures/mnist_test.csv"
    )
    output_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/output_bitmap_mnist.png"
    )

    # Instanciate Hishiryo
    HishiryoConverter = Hishiryo.Hishiryo()
    assert HishiryoConverter.convertCSVToBitmap(
        input_path,
        output_path,
    )
