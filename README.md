# hishiryo (alpha)

Convert a csv dataset into a radial pixel map

This little experiment aims at trying to quickly represent the content of a csv table.
Each datapoint (like a cell in an excel sheet) is converted into a pixel , and this pixel is diplayed on a circular graph.
This version supports the following data formats : float, integers, Text as categories.

### How to install

    #PyPI
    pip install hishiryo

### Dependencies

-   Pandas
-   CV2

### How to use

1 - find an appropriate csv. (more than 1000 rows will not render very well)

2 - in python 3 :

    from hishiryo import Hishiryo

    HishiryoConverter = Hishiryo.Hishiryo()

    HishiryoConverter.convertCSVToRadialBitmap(input_path,separator,output_path,radius,None,"Dot")


function convertCSVToRadialBitmap(input_path,separator,output_path,radius,sort_by,glyph_type)
-   `input path` is the path to your csv file (e.g. /home/user/iris.csv)
-   `output path` is the path to your target image file (e.g. /home/user/iris.png) The fileformat you want is autodetected thanks to CV2 functionalities.
-   `separator` is the character separator in your csv (e.g. ",")
-   `radius` (in pixel) is the size of the radius of the disk where the pixels will be drawn. The higher it is the bigger and sharper your output image will be. (e.g:  1500)'
-   `sort_by` is the name of the column or the list of column you want to sort you data. (e.g. "Sepal.Length", or ["Sepal.Length","Sepal.Width"])
-   `glyph_type` is the type of representation you want for the pixels. it can be one among the following : "Dot","Square" or "Polygon"

### Licence

GNU General Public License v3.0

#note
Sorry for the crappy code, i will improve it with the time on the use of the package! If you have any feedback or comment on the use of this little experiment, feel free to tell me!

