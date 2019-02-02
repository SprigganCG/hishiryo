import math
import random
from collections import defaultdict
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import svgwrite
import cv2

class Hishiryo:

    def __init__(self):
        ''' Constructor for this class. '''
        # Set Version and default parameters
        self.VERSION = "0.1.1"
        self.config_background_color = (0, 0, 0)
        self.config_image_size_multiplyer = 1
        self.radial_render_radius = 2000
        self.radial_render_outer_padding = int(self.radial_render_radius * 0.05)
        self.radial_render_inner_padding = int(self.radial_render_radius * 0.4)
        self.radial_render_circle_radius = 0.2

    def getVersion(self):
        return self.VERSION

    def get_radial_coordinates(self,x_center, y_center, inner_padding, dataset_column_count, dataset_row_count,
                               current_column, current_row,disc_radius):
        """
        Compute coordinates of a datapoint for radial representation

        :param x_center: the x coordinates of the center of the radial representation bitmap
        :param y_center: the y coordinates of the center of the radial representation bitmap
        :param inner_padding: distance between the center of the radial representation bitmap and the first datapoint row
        :param dataset_column_count:
        :param dataset_row_count:
        :param current_column:
        :param current_row:
        :param disc_radius:
        :return: tuple containing the target position of the datapoint on the radial representation.
        """

        #  define the elevation of the pixel based on it's column in the dataset (if first column, the outer it will be
        elevation = ((current_column / dataset_column_count) * disc_radius) + inner_padding
        initial_coordinates = (0, elevation)

        #  define the rotation of the coordinates based on the location of the datapoint in the rows
        rotation_angle = (current_row / dataset_row_count) * (math.pi * 2)

        #  preform rotation of coordinates with an angle = to rotation angle
        new_coordinates = (
        initial_coordinates[0] * math.cos(rotation_angle) - initial_coordinates[1] * math.sin(rotation_angle),
        initial_coordinates[0] * math.sin(rotation_angle) + initial_coordinates[1] * math.cos(
            rotation_angle))

        # x' = x*cos(angle) - y*sin(angle)
        # y' = x*sin(angle) + y*cos(angle)

        radial_coordinates = (new_coordinates[0] + x_center, new_coordinates[1] + y_center)

        return radial_coordinates

    def get_square_radial_coordinates(self, x_center, y_center, inner_padding, dataset_column_count, dataset_row_count,
                                      current_column, current_row, disc_radius,glyph_size):

        # Properties of the square
        square_height = glyph_size[0]
        square_width = glyph_size[1]

        #  define the elevation of the glyph based on it's column in the dataset (if first column, the outer it will be
        elevation = ((current_column / dataset_column_count) * disc_radius) + inner_padding
        initial_coordinates = (0, elevation)

        #  the square has 4 points , 1,2,3,4
        initial_coordinates = []
        initial_coordinates.append((square_width / 2, elevation + (square_height / 2)))
        initial_coordinates.append((-square_width / 2, elevation + (square_height / 2)))
        initial_coordinates.append((-square_width / 2, elevation - (square_height / 2)))
        initial_coordinates.append((square_width / 2, elevation - (square_height / 2)))

        # these square points are rotated depending on the location of the datapoint.

        #  define the rotation of the coordinates based on the location of the datapoint in the rows
        rotation_angle = (current_row / dataset_row_count) * (math.pi * 2)

        #  preform rotation of coordinates with an angle = to rotation angle
        new_coordinates = []
        for current_coordinates in initial_coordinates:
            new_coordinates.append(
                (current_coordinates[0] * math.cos(rotation_angle) - current_coordinates[1] * math.sin(rotation_angle),
                 current_coordinates[0] * math.sin(rotation_angle) + current_coordinates[1] * math.cos(rotation_angle)))

        # x' = x*cos(angle) - y*sin(angle)
        # y' = x*sin(angle) + y*cos(angle)

        radial_coordinates = []
        for current_radial_coordinates in new_coordinates:
            radial_coordinates.append((current_radial_coordinates[0] + x_center, current_radial_coordinates[1] + y_center))

        return radial_coordinates

    def get_polygon_radial_coordinates(self, x_center, y_center, inner_padding, dataset_column_count, dataset_row_count,
                                      current_column, current_row, disc_radius,glyph_size):

        bitmap_width = dataset_column_count
        bitmap_height = dataset_row_count

        #define the distance between two rows on the disc
        interrow_distance = (disc_radius-inner_padding)/(bitmap_width)
        #print("interrow distance:",interrow_distance)

        # the polygon will be made of 4 points.
        #compute the distance between the two higher points
        #radius @ height + 1/2 interrow distance.
        circle_radius = (((current_column / bitmap_width) * disc_radius) + inner_padding) + (0.5*interrow_distance)
        current_row_perimeter = 2 * math.pi * circle_radius
        inter_glyph_distance_top = (current_row_perimeter / bitmap_height)

        #compute the distance between the two lower points
        #radius @ height - 1/2 interrow distance.
        circle_radius = (((current_column / bitmap_width) * disc_radius) + inner_padding) - (0.5*interrow_distance)
        current_row_perimeter = 2 * math.pi * circle_radius
        inter_glyph_distance_bottom = (current_row_perimeter / bitmap_height)


        # Properties of the square
        square_height = interrow_distance
        square_width_top = inter_glyph_distance_top
        square_width_bottom = inter_glyph_distance_bottom

        #  define the elevation of the glyph based on it's column in the dataset (if first column, the outer it will be
        elevation = ((current_column / dataset_column_count) * disc_radius) + inner_padding
        initial_coordinates = (0, elevation)

        #  the square has 4 points , 1,2,3,4
        initial_coordinates = []
        initial_coordinates.append((square_width_top / 2, elevation + (square_height / 2)))
        initial_coordinates.append((-square_width_top / 2, elevation + (square_height / 2)))
        initial_coordinates.append((-square_width_bottom / 2, elevation - (square_height / 2)))
        initial_coordinates.append((square_width_bottom / 2, elevation - (square_height / 2)))

        # these square points are rotated depending on the location of the datapoint.

        #  define the rotation of the coordinates based on the location of the datapoint in the rows
        rotation_angle = (current_row / dataset_row_count) * (math.pi * 2)

        #  perform rotation of coordinates with an angle = to rotation angle
        new_coordinates = []
        for current_coordinates in initial_coordinates:
            new_coordinates.append(
                (current_coordinates[0] * math.cos(rotation_angle) - current_coordinates[1] * math.sin(rotation_angle),
                 current_coordinates[0] * math.sin(rotation_angle) + current_coordinates[1] * math.cos(rotation_angle)))

        # x' = x*cos(angle) - y*sin(angle)
        # y' = x*sin(angle) + y*cos(angle)

        radial_coordinates = []
        for current_radial_coordinates in new_coordinates:
            radial_coordinates.append((current_radial_coordinates[0] + x_center, current_radial_coordinates[1] + y_center))

        return radial_coordinates


    def processDataFrame2Bitmap(self, dataset_df):

        #  2 Read the dataset specs
        print("Number of rows", dataset_df.shape[0])
        print("Number of columns", dataset_df.shape[1])

        #  3 define the size of the bitmap
        bitmap_width = dataset_df.shape[1]
        bitmap_height = dataset_df.shape[0]
        datapoint_count = bitmap_height * bitmap_width

        print("Datapoints to display:", datapoint_count)

        #  4 initialize the image in memory
        print("create image")
        current_bitmap_opencv = np.zeros((bitmap_height,bitmap_width, 3), np.uint8)

        #  5 process each column data, and fill the bitmap with dynamic colors

        #  for each dataset column, detect the id and the type. if float, then populate the corresponding pixels on the bitmap applying a simple normalisation.
        for id, current_column in enumerate(dataset_df.columns):

            print(id, current_column)

            #if column is empty, fill it with 0
            if len(dataset_df[current_column].value_counts()) == 0:
                print('empty column')
                dataset_df[current_column] = 0

            if dataset_df[current_column].dtypes == 'float64' or dataset_df[current_column].dtypes == 'int64':

                if dataset_df[current_column].dtypes == 'float64':
                    colorTint = (0.5, 0.5, 1.0) # red for floats
                if dataset_df[current_column].dtypes == 'int64':
                    colorTint = (1.0, 0.5, 0.5) # blue for integers

                #  get the min, max
                column_min = dataset_df[current_column].min()
                column_max = dataset_df[current_column].max()

                if dataset_df[current_column].dtypes == 'float64':
                    dataset_df[current_column].fillna(0.0, inplace=True)
                if dataset_df[current_column].dtypes == 'int64':
                    dataset_df[current_column].fillna(0, inplace=True)

                #  get the dataseries

                if(column_max - column_min)==0: column_data = (dataset_df[current_column])
                else: column_data = ((dataset_df[current_column] - column_min) / (column_max - column_min))
                column_data_rgb = [(int(round(x * 255 * colorTint[0])), int(round(x * 255 * colorTint[1])),
                                    int(round(x * 255 * colorTint[2]))) for x in column_data]

                # write the pixels
                for current_pixel in range(0, bitmap_height):
                    current_bitmap_opencv[current_pixel,id] = [column_data_rgb[current_pixel][2],column_data_rgb[current_pixel][1],column_data_rgb[current_pixel][0]]

            elif dataset_df[current_column].dtypes == 'object':

                colorTint = (0.5, 1.0, 0.5)

                #  remove the nan and replace it with zero. (it's not an object for sure..)
                dataset_df[current_column].fillna(0, inplace=True)
                #  get the modalities.
                column_labels = set(dataset_df[current_column].values.tolist())

                # attribute colors to each modality
                modalities_color_dict = defaultdict()
                for current_label in column_labels:
                    modalities_color_dict[current_label] = (
                    random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                #  get the dataseries
                dataset_df[current_column].fillna(0, inplace=True)
                column_data = dataset_df[current_column].values.tolist()
                column_data_rgb = [
                    (modalities_color_dict[x][0], modalities_color_dict[x][1], modalities_color_dict[x][2]) for x in
                    column_data]

                # write the pixels
                for current_pixel in range(0, bitmap_height):
                    current_bitmap_opencv[current_pixel,id] = [column_data_rgb[current_pixel][2],
                                                                column_data_rgb[current_pixel][1],
                                                                column_data_rgb[current_pixel][0]]
            else:
                print("Column type unknown")



        return current_bitmap_opencv

    def convertCSVToBitmap(self,input_dataset_path,output_image_path):

        #load the dataset
        print("Load")
        dataset_df = pd.read_csv(input_dataset_path)

        #convert it to bitmap
        print("Convert")
        current_bitmap = Hishiryo.processDataFrame2Bitmap(self, dataset_df)

        # write bitmap on disk
        print("Write")
        cv2.imwrite(output_image_path, current_bitmap)

        return True

    def convertCSVToRadialSVG(self,input_dataset_path,output_image_radial_svg_render_path):

        #load the dataset
        print("Load")
        dataset_df = pd.read_csv(input_dataset_path)

        #convert it to bitmap
        print("Convert")
        current_bitmap = Hishiryo.processDataFrame2Bitmap(self, dataset_df)

        bitmap_width = dataset_df.shape[1]
        bitmap_height = dataset_df.shape[0]

        #render a second image (svg) as a radial representation

        # compute origin coordinates
        radial_render_width = self.radial_render_radius * 2 + self.radial_render_outer_padding * 2 + self.radial_render_inner_padding * 2
        radial_render_height = self.radial_render_radius * 2 + self.radial_render_outer_padding * 2 + self.radial_render_inner_padding * 2
        radial_render_origin_coordinates = (int(round(radial_render_width / 2)), int(round(radial_render_height / 2)))


        #  initialize the rendered svg

        # proceed to drawing
        svg_document = svgwrite.Drawing(filename=output_image_radial_svg_render_path,
                                        size=(radial_render_width, radial_render_height), debug=True,
                                        profile='full')

        # fill background in black with a rect
        svg_document.add(
            svg_document.rect(insert=(0, 0), size=(radial_render_width, radial_render_height), stroke_width="0",
                              stroke="black", fill="rgb(" + str(self.config_background_color[0]) + "," + str(
                    self.config_background_color[1]) + "," + str(self.config_background_color[2]) + ")"))

        #  parse all the pixels of the bitmap in memory and map them to the radial svg image

        #  transpose to manage more easily the bitmap data (a dataset often have less column than rows, the radial representation should put columnar data as contour of a circle
        for current_pixel_row in range(0, bitmap_width):

            print("defining render picture row", current_pixel_row)
            # compute disc radius
            # compute the perimeter of the circle, check how many datapoint on the circle, divide to obtain de diameter, divide by two for the radius
            radial_render_circle_radius = Hishiryo.computeDotGlyphRadius(self, current_pixel_row, bitmap_width,
                                                                         bitmap_height, self.radial_render_radius,
                                                                         self.radial_render_inner_padding)
            print("Radius Estimated:", radial_render_circle_radius)

            for current_pixel_column in range(0, bitmap_height):
                # print("parsing picture column",current_pixel_column)
                current_radial_coordinates = Hishiryo.get_radial_coordinates(self,radial_render_origin_coordinates[0],
                                                                    radial_render_origin_coordinates[1],
                                                                    self.radial_render_inner_padding, bitmap_width,
                                                                    bitmap_height, current_pixel_row,
                                                                    current_pixel_column, self.radial_render_radius)
                # get pixel color
                pixel_color = current_bitmap[current_pixel_column,current_pixel_row]
                # print(pixel_color)

                #  create a disc for each pixel.
                svg_document.add(
                    svg_document.circle(center=(current_radial_coordinates[0], current_radial_coordinates[1]),
                                        r=radial_render_circle_radius,
                                        fill="rgb(" + str(pixel_color[0]) + "," + str(pixel_color[1]) + "," + str(
                                            pixel_color[2]) + ")"))

        #  8 write image to disk
        print("write radial svg image")
        svg_document.save()

        return True

    def convertCSVToRadialBitmap(self,input_dataset_path,separator,output_image_radial_opencv_render_path,radius=None,sort_by=None,glyph_type="Dot"):


        #load the dataset
        print("Load")
        dataset_df = pd.read_csv(input_dataset_path,sep=separator)

        #check if a radius is defined by the user
        if radius is None:
            pass
        else :
            self.radial_render_radius = radius
            self.radial_render_outer_padding = int(self.radial_render_radius * 0.05)
            self.radial_render_inner_padding = int(self.radial_render_radius * 0.4)
            self.radial_render_circle_radius = 0.2

        #check if a sort is required by user
        if sort_by is None: pass
        else :
            dataset_df.sort_values(by=sort_by, ascending=False, na_position='first',inplace=True)

        #convert it to bitmap
        print("Convert")
        current_bitmap = Hishiryo.processDataFrame2Bitmap(self, dataset_df)

        bitmap_width = dataset_df.shape[1]
        bitmap_height = dataset_df.shape[0]

        # render a second image as a radial

        # compute origin coordinates
        radial_render_width = self.radial_render_radius * 2 + self.radial_render_outer_padding * 2 + self.radial_render_inner_padding * 2
        radial_render_height = self.radial_render_radius * 2 + self.radial_render_outer_padding * 2 + self.radial_render_inner_padding * 2
        radial_render_origin_coordinates = (int(round(radial_render_width / 2)), int(round(radial_render_height / 2)))

        #  initialize the rendered opencv bitmap
        opencv_image = np.zeros((radial_render_width, radial_render_height, 3), np.uint8)
        #self.config_background_color
        opencv_image[:,:,0] = self.config_background_color[0]
        opencv_image[:,:,1] = self.config_background_color[1]
        opencv_image[:,:,2] = self.config_background_color[2]


        print("Prepare Radial OpenCV rendering", radial_render_width, radial_render_height,
              radial_render_origin_coordinates)

        #  parse all the pixels of the bitmap in memory and map them to the radial svg image

        #  transpose to manage more easily the bitmap data (a dataset often have less column than rows, the radial representation should put columnar data as contour of a circle
        for current_pixel_row in range(0, bitmap_width):

            print("Render CSV Column:", current_pixel_row)

            # compute glyph radius if we use dot glyph
            if glyph_type == "Dot":
                # compute the perimeter of the circle, check how many datapoint on the circle, divide to obtain de diameter, divide by two for the radius of the glyph
                radial_render_circle_radius = Hishiryo.computeDotGlyphRadius(self, current_pixel_row, bitmap_width, bitmap_height, self.radial_render_radius, self.radial_render_inner_padding)
                print("Glyph Estimated:", radial_render_circle_radius)

            # compute glyph size (witdh & weight) if we use dot Square
            if glyph_type == "Square":
                # compute the perimeter of the circle, check how many datapoint on the circle, divide to obtain de size, of the glyph
                glyph_size = Hishiryo.computeSquareGlyphSize(self, current_pixel_row, bitmap_width, bitmap_height, self.radial_render_radius, self.radial_render_inner_padding)
                print("Glyph Size:", glyph_size)

            # compute glyph size (witdh & weight) if we use dot Square
            if glyph_type == "Polygon":
                # compute the perimeter of the circle, check how many datapoint on the circle, divide to obtain de size, of the glyph
                glyph_shape = Hishiryo.computePolygonGlyphShape(self, current_pixel_row, bitmap_width, bitmap_height, self.radial_render_radius, self.radial_render_inner_padding)
                print("Glyph Size:", glyph_shape)



            for current_pixel_column in range(0, bitmap_height):

                #for glyph = dot (other glyph could be square,triangles etc...

                if glyph_type == "Dot":

                    # print("parsing picture column",current_pixel_column)
                    current_radial_coordinates = Hishiryo.get_radial_coordinates(self,radial_render_origin_coordinates[0],
                                                                                 radial_render_origin_coordinates[1],
                                                                                 self.radial_render_inner_padding, bitmap_width,
                                                                                 bitmap_height, current_pixel_row,
                                                                                 current_pixel_column, self.radial_render_radius)
                    # get pixel color
                    pixel_color = current_bitmap[current_pixel_column,current_pixel_row]
                    # print(pixel_color)

                    #  create a disc for each pixel.
                    cv2.circle(opencv_image, (int(current_radial_coordinates[0]), int(current_radial_coordinates[1])),
                               int(radial_render_circle_radius), (int(pixel_color[2]), int(pixel_color[1]), int(pixel_color[0])),
                               thickness=-1, lineType=8, shift=0)

                if glyph_type == "Square":

                    radial_render_circle_radius = 1
                    # print("parsing picture column",current_pixel_column)
                    current_square_radial_coordinates = Hishiryo.get_square_radial_coordinates(self,radial_render_origin_coordinates[0],
                                                                                 radial_render_origin_coordinates[1],
                                                                                 self.radial_render_inner_padding, bitmap_width,
                                                                                 bitmap_height, current_pixel_row,
                                                                                 current_pixel_column, self.radial_render_radius,glyph_size)

                    # get pixel color
                    pixel_color = current_bitmap[current_pixel_column,current_pixel_row]
                    # print(pixel_color)

                    current_square_radial_coordinates = np.array(current_square_radial_coordinates,np.int32)
                    #  create a square for each pixel.
                    #for sub_current_square_radial_coordinates in current_square_radial_coordinates:
                    #    cv2.circle(opencv_image, (int(sub_current_square_radial_coordinates[0]), int(sub_current_square_radial_coordinates[1])),
                    #               int(radial_render_circle_radius), (int(pixel_color[2]), int(pixel_color[1]), int(pixel_color[0])),
                    #               thickness=1, lineType=8, shift=0)
                    cv2.fillPoly(opencv_image, [current_square_radial_coordinates],(int(pixel_color[2]), int(pixel_color[1]), int(pixel_color[0])),lineType=0,shift=0)

                if glyph_type == "Polygon":


                    radial_render_circle_radius = 1
                    # print("parsing picture column",current_pixel_column)
                    current_square_radial_coordinates = Hishiryo.get_polygon_radial_coordinates(self,radial_render_origin_coordinates[0],
                                                                                 radial_render_origin_coordinates[1],
                                                                                 self.radial_render_inner_padding, bitmap_width,
                                                                                 bitmap_height, current_pixel_row,
                                                                                 current_pixel_column, self.radial_render_radius,glyph_shape)

                    # get pixel color
                    pixel_color = current_bitmap[current_pixel_column,current_pixel_row]
                    # print(pixel_color)

                    current_square_radial_coordinates = np.array(current_square_radial_coordinates,np.int32)
                    #  create a square for each pixel.
                    #for sub_current_square_radial_coordinates in current_square_radial_coordinates:
                    #    cv2.circle(opencv_image, (int(sub_current_square_radial_coordinates[0]), int(sub_current_square_radial_coordinates[1])),
                    #               int(radial_render_circle_radius), (int(pixel_color[2]), int(pixel_color[1]), int(pixel_color[0])),
                    #               thickness=1, lineType=8, shift=0)
                    cv2.fillPoly(opencv_image, [current_square_radial_coordinates],(int(pixel_color[2]), int(pixel_color[1]), int(pixel_color[0])),lineType=0,shift=0)


        #  8 write images to disk
        print("write radial opencv image")
        #out_resized = cv2.resize(opencv_image,None, fx=0.5, fy=0.5,interpolation = cv2.INTER_AREA)
        cv2.imwrite(output_image_radial_opencv_render_path, opencv_image)

        return True

    def computeDotGlyphRadius(self, current_pixel_row, bitmap_width, bitmap_height, radial_render_radius, radial_render_inner_padding):

        #compute the radius of a datapoint
        circle_radius = ((current_pixel_row / bitmap_width) * radial_render_radius) + radial_render_inner_padding
        current_row_perimeter = 2 * math.pi * circle_radius
        radial_render_circle_radius = (current_row_perimeter / bitmap_height) / 2

        #compute the distance between two circles
        interrow_distance = (radial_render_radius-radial_render_inner_padding)/(bitmap_width)
        interrow_distance = interrow_distance /2 #divide by two as two datapoint circles share the distance

        #if the radius must not be superior to the distance to prevent overlap
        if radial_render_circle_radius > interrow_distance :  radial_render_circle_radius = interrow_distance

        return radial_render_circle_radius

    def computeSquareGlyphSize(self, current_pixel_row, bitmap_width, bitmap_height, radial_render_radius, radial_render_inner_padding):

        #compute the size of a square datapoint

        circle_radius = ((current_pixel_row / bitmap_width) * radial_render_radius) + radial_render_inner_padding
        current_row_perimeter = 2 * math.pi * circle_radius
        inter_glyph_distance = (current_row_perimeter / bitmap_height) / 2
        glyph_width = inter_glyph_distance
        glyph_height = glyph_width

        glyph_size = [glyph_width,glyph_height]

        interrow_distance = (radial_render_radius-radial_render_inner_padding)/(bitmap_width) / 2
        print(interrow_distance)

        #the height must not be superior to the distance to prevent overlap
        if glyph_height > interrow_distance :
            glyph_size[0] = interrow_distance
            glyph_size[1] = interrow_distance

        return glyph_size

    def computePolygonGlyphShape(self, current_pixel_row, bitmap_width, bitmap_height, radial_render_radius, radial_render_inner_padding):

        #compute the position of each vertex of the polygon

        #define the distance between two rows on the disc
        interrow_distance = (radial_render_radius-radial_render_inner_padding)/(bitmap_width)
        print(interrow_distance)


        circle_radius = ((current_pixel_row / bitmap_width) * radial_render_radius) + radial_render_inner_padding
        current_row_perimeter = 2 * math.pi * circle_radius
        inter_glyph_distance = (current_row_perimeter / bitmap_height)
        glyph_width = inter_glyph_distance
        glyph_height = glyph_width

        glyph_size = [glyph_width,glyph_height]


        #the height must not be superior to the distance to prevent overlap
        if glyph_height > interrow_distance :
            glyph_size[0] = interrow_distance
            glyph_size[1] = interrow_distance
        return glyph_size




