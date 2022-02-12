"""Integration tests for the sort option."""
import pytest
from hishiryo import Hishiryo
import os


# simple run against a test csv dataset and generate a
# simple bitmap
def test_render_bitmap_from_csv():

    # define test case
    input_path = os.path.dirname(os.path.realpath(__file__)) \
        + "/../fixtures/mnist_test.csv"
    output_path = os.path.dirname(
        os.path.realpath(__file__)) + "/output_bitmap_mnist.png"

    # Instanciate Hishiryo
    HishiryoConverter = Hishiryo.Hishiryo()
    assert HishiryoConverter.convertCSVToBitmap(input_path,
                                                output_path,
                                                )


