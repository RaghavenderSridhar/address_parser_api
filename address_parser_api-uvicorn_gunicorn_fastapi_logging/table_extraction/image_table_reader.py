import cv2
import json
import numpy as np
import pandas as pd
import csv
import os
import logging
import configparser
import argparse
import datetime
import time
import os.path
import pytesseract
import math
import linecache
import sys
import re
import bs4
import copy

class ImageTableBorderReader:
    """
    This is a class to convert table with border information in images to excel file.
    """

    def __init__(self, pytesseract_path, temp_folderpath, logger, debug_mode_check):
        """
        The constructor for image_table_border_detection class.
        """
        self.pytesseract_path = pytesseract_path
        self.logger = logger
        self.debug_mode_check = debug_mode_check
        self.temp_folderpath = temp_folderpath
        pytesseract.pytesseract.tesseract_cmd = pytesseract_path

    def extract(self, img, img_name,  save_folder_path):
        """
        The function to extract information from the images.

        Parameters:
            img : image
            img_name: image file name
            save_folder_path: Path to save the excel file
        Returns:
        box, bitnot = self._detect_all_cells(img, img_name)
            json response of the extracted table
            error
        """
        img_arr = copy.deepcopy(img)
        process_2_reqd = False
        try:
            box, warning = self._detect_all_cells(
                img_arr, img_name, process_2_reqd=False, process_3_reqd=True)
        except:
            try:
                box, warning = self._detect_all_cells(img_arr, img_name)
            except:
                try:
                    box, process_2_reqd, mean_word_h, warning = self._second_run_extraction_processing(
                        img_arr, img_name)
                except Exception as ex:
                    raise Exception(str(ex))

        mean = sum([box[i][3] for i in range(len(box))])//len(box)
        box.sort(key=lambda x: (x[1], x[0]))
        row, column = [], []
        for i in range(len(box)):
            if(i == 0):
                column.append(box[i])
                previous = box[i]
            else:
                if(box[i][1] == previous[1]):
                    column.append(box[i])
                    previous = box[i]
                else:
                    row.append(column)
                    column = []
                    previous = box[i]
                    column.append(box[i])
        row.append(column)

        self.logger.info("column: {}".format(column))
        self.logger.info("row: {}".format(row))

        if row[0][0][3] < mean//5:
            row.remove(row[0])

        if(process_2_reqd == True):
            # remove rows where row height is less than word height
            remove_row = []
            for i in range(len(row)):
                if row[i][0][3] < mean_word_h - mean_word_h/4:
                    remove_row.append(row[i])
            row = [x for x in row if x not in remove_row]

        countcol, index = len(row[0]), 0
        for i in range(len(row)):
            if countcol < len(row[i]):
                index = i
                countcol = len(row[i])
        i, j = index, 0
        center = [int(row[i][j][0]+row[i][j][2]/2)
                  for j in range(len(row[i])) if row[0]]

        center = np.array(center)
        center.sort()
        self.logger.info("center: {}".format(center))

        finalboxes = []
        for i in range(len(row)):
            lis = []
            for k in range(countcol):
                lis.append([])
            for j in range(len(row[i])):
                diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(row[i][j])
            finalboxes.append(lis)

        outer, false_r_count = [], 0
        for i in range(len(finalboxes)):
            count = 0
            for j in range(len(finalboxes[i])):
                inner = ''
                if(len(finalboxes[i][j]) == 0):
                    outer.append(' ')
                else:
                    for k in range(len(finalboxes[i][j])):
                        y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][
                            k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                        finalimg = img[x:x+h, y:y+w]
                        if(len(finalimg.shape) == 3):
                            finalimg = cv2.cvtColor(
                                finalimg, cv2.COLOR_BGR2GRAY)
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(
                            finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                        resizing = cv2.resize(
                            border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        dilation = cv2.dilate(resizing, kernel, iterations=2)
                        erosion = cv2.erode(dilation, kernel, iterations=2)
                        if(self.debug_mode_check == True):
                            cv2.imwrite(
                                f'{self.temp_folderpath}\\{img_name}_cell_{i}_{j}.png', erosion)
                        out = pytesseract.image_to_string(
                            finalimg, config='--psm 3')
                        out = out.strip()
                        if(len(out) == 0):
                            out = pytesseract.image_to_string(
                                finalimg, config='--psm 6')
                        if(out == ''):
                            count += 1
                        if(out == ''):
                            out = pytesseract.image_to_string(
                                finalimg, config='--psm 7')
                        inner = inner + out.strip()
                        inner = ' ' if inner == '' else inner
                        outer.append(inner)
            if(count == len(finalboxes[i])):
                outer = outer[0:len(outer)-count]
                false_r_count += 1
        arr = np.array(outer)
        header_arr = arr[:countcol].tolist()
        for i, header in enumerate(header_arr):
            j = 1
            for k, header_2 in enumerate(header_arr):
                if(i != k):
                    if header == header_2 and header != ' ':
                        header_arr[k] = header_2+str(j)
                        j += 1
                    elif header == header_2 and header_2 == ' ':
                        header_arr[k] = "col_" + str(j+1)
                        j += 1
            if(header == ' '):
                header_arr[i] = "col_1"
        body_arr = arr[countcol:]
        error = None
        try:
            dataframe = pd.DataFrame(body_arr.reshape(
                len(row)-1-false_r_count, countcol), columns=header_arr)
        except Exception as e:
            error = e
            body_arr, col_count, row_count = self._array_reshape_fix(
                body_arr, countcol, len(row)-1-false_r_count)
            body_arr = np.array(body_arr)
            dataframe = pd.DataFrame(body_arr.reshape(
                row_count, col_count), columns=header_arr)
        response = json.loads(dataframe.to_json(orient='records'))
        if(len(response) == 0):
            response = header_arr
        output_dataframe = dataframe.style.set_properties(align="left")
        self.logger.info("showing data: {}".format(dataframe))
        if(save_folder_path != None):
            output_dataframe.to_excel(save_folder_path + '.xlsx')
        return response, error, warning

    def _detect_all_cells(self, img, img_name, process_2_reqd=False, process_3_reqd=False):
        try:
            warning = []
            # # check orientation and rotate accordingly
            # angle = 360-int(re.search('(?<=Rotate: )\d+',
            #                           pytesseract.image_to_osd(img)).group(0))
            # img = self._rotate_image(img, angle)
            if(len(img.shape) == 3):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(process_2_reqd == True or process_3_reqd == True):
                img_bin = cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 35)  # 30
                img_bin = cv2.GaussianBlur(img_bin, (1, 1), 0)
            else:
                thresh, img_bin = cv2.threshold(
                    img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_bin = 255-img_bin
            if(self.debug_mode_check == True):
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'img_bin.png', img_bin)

            _, actual_h_grouped = self._detect_horizontal_lines(
                img, img_bin, img_name, 1, -1, process_2_reqd=process_2_reqd)
            if(len(actual_h_grouped) == 0):
                raise Exception(
                    "No cells detected: Horizontal Lines not detected")
            start = 2 if len(actual_h_grouped) > 3 else 0
            px_diff_arr = [actual_h_grouped[i][-1][1] -
                           actual_h_grouped[i][0][1] for i in range(start, len(actual_h_grouped))]
            px_diff = round(sum(px_diff_arr)/len(px_diff_arr))
            px_diff = px_diff-1 if px_diff > 0 else px_diff
            angle = math.degrees(math.atan(px_diff/img.shape[1]))
            angle_dir_arr = [actual_h_grouped[i][-1][0] > actual_h_grouped[i][0][0]
                             for i in range(start, len(actual_h_grouped))]
            count = 0
            for vote in angle_dir_arr:
                if(vote == True):
                    count += 1
            angle = angle if(count > len(angle_dir_arr)/2) else -angle
            if(angle > 0.1):
                warning = "Skew detected: "+str(angle)
            # rotated = self._deskew_image(img_bin, angle)
            # if(self.debug_mode_check == True):
            #     cv2.imwrite(self.temp_folderpath+"\\" +
            #                 img_name+'rotated.png', rotated)

            # img = self._deskew_image(img, angle)
            if(process_2_reqd == True or process_3_reqd == True):
                horizontal_lines, actual_h_grouped = self._detect_horizontal_lines(
                    img, img_bin, img_name, 3, -1, process_2_reqd=process_2_reqd)

                vertical_lines, _ = self._detect_vertical_lines(
                    img, img_bin, img_name, 3, -1, process_2_reqd=process_2_reqd)
            else:
                horizontal_lines, actual_h_grouped = self._detect_horizontal_lines(
                    img, img_bin, img_name, process_2_reqd=process_2_reqd)

                vertical_lines, _ = self._detect_vertical_lines(
                    img, img_bin, img_name, process_2_reqd=process_2_reqd)

            # adds the vertical and horizontal lines images
            img_vh = cv2.addWeighted(
                vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

            thresh, img_vh = cv2.threshold(
                img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # bitxor = cv2.bitwise_xor(img, img_vh)
            # bitnot = cv2.bitwise_not(bitxor)

            if(self.debug_mode_check == True):
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'img_vh.png', img_vh)

            contours, hierarchy = cv2.findContours(
                img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.logger.info("Detect contours: {}".format(contours, hierarchy))

            box = []

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if (w <= img.shape[1]-img.shape[1]/5 and w > img.shape[1]/30 and h <= img.shape[1]-img.shape[0]/5 and h >= img.shape[0]/50):
                    image = cv2.rectangle(
                        img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    box.append([x, y, w, h])

            if(self.debug_mode_check == True):
                img_copy = img.copy()
                # highlight
                for b in box:
                    img_copy = cv2.rectangle(
                        img_copy, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), color=(255, 100, 0), thickness=1)
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'_boxes.png', img_copy)
            if(len(box) == 0):
                raise Exception("No cells detected")
            return box, warning

        except Exception as ex:
            # self._second_run_extraction_processing(img, img_name)
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno
            raise Exception(str(ex)+" ((Line: "+str(lineno)+")")

    def _detect_vertical_lines(self, img, img_bin, img_name, MAXGAP=3, COMPARE_POS=0, process_2_reqd=False):
        try:
            # image with all vertical lines
            if(img.shape[0] <= 100):
                ver_kernel_len = 3
            elif(img.shape[0] <= 150):
                ver_kernel_len = np.array(img).shape[0]//25
            elif(img.shape[0] <= 500):
                ver_kernel_len = np.array(img).shape[0]//50
            elif(img.shape[0] <= 1500):
                ver_kernel_len = np.array(img).shape[0]//100
            else:
                ver_kernel_len = np.array(img).shape[0]//200
            ver_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, ver_kernel_len))
            if(process_2_reqd == True):
                image_1 = cv2.dilate(img_bin, ver_kernel, iterations=1)
                vertical_lines = cv2.erode(image_1, ver_kernel, iterations=1)
            else:
                image_1 = cv2.erode(img_bin, ver_kernel, iterations=2)
                vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=2)
            if(self.debug_mode_check == True):
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'image_v1.png', image_1)

            h_points = []
            # # vertical lines coordinate detection
            MIN_LINE_SCALE, threshold, minLineLength, maxLineGap = 0.4, 10, img.shape[
                0]//50, img.shape[1]//80
            rotated = self._rotate_image(vertical_lines, 90)
            edges = cv2.Canny(rotated, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if(x2-x1 > 0 and y1 == y2):
                        h_points.append([x1, y1, x2, y2, x2-x1])
            h_points = sorted(h_points, key=lambda x: (x[1], x[0]))
            # Clustering lines into one with max pixel diff of 3
            if(len(h_points) == 0):
                raise Exception(
                    "No cells detected: Horizontal Lines not detected")
            h_grouped = self._cluster(h_points, MAXGAP, 1, 2, 0, COMPARE_POS)
            horizontal_lines = np.zeros((img.shape[1], img.shape[0]), np.uint8)
            horizontal_lines.fill(255)
            actual_h_points, actual_h_grouped = [], []
            # for each clustered line, finds the total width of line
            # and if greater than 40% of total image width it draws a complete line on the horizontal_lines image
            for group in h_grouped:
                line_height, count = 0, 0
                for line in group:
                    # if any line is greater than 10% of the image width, then consider it as potential line
                    if(line[4] > 0.05*img.shape[0]):
                        count += 1
                    line_height += line[4]
                if(line_height >= MIN_LINE_SCALE * img.shape[0] and count > 0):
                    actual_h_grouped.append(group)
                    actual_h_points.append(
                        [0, group[0][1], img.shape[0], group[0][3]])
                    cv2.line(
                        horizontal_lines, (0, group[0][1]), (img.shape[0], group[0][3]), (0, 0, 0), 2)
            vertical_lines = self._rotate_image(horizontal_lines, 270)

            # v_points = []
            # edges = cv2.Canny(vertical_lines, 50, 150, apertureSize=3)

            # lines = cv2.HoughLinesP(
            #     edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
            # for line in lines:
            #     for x1, y1, x2, y2 in line:
            #         if y1-y2 > 0 and x1 == x2:
            #             v_points.append([x1, y1, x2, y2, y1-y2])
            # v_points = sorted(v_points, key=lambda x: (x[0], x[1]))
            # if(len(v_points) == 0):
            #     raise Exception(
            #         "No cells detected: Vertical Lines not detected")
            # # Clustering lines into one with max pixel diff of 3
            # v_grouped = self._cluster(v_points, MAXGAP, 0, 1, 3, COMPARE_POS)
            # vertical_lines = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            # vertical_lines.fill(255)
            # actual_v_points, actual_v_grouped = [], []
            # # for each clustered line, finds the total height of line
            # # and if greater than 40% of total image height it draws a complete line on the vertical_lines image
            # for group in v_grouped:
            #     line_height = 0
            #     for line in group:
            #         line_height += line[4]
            #     if(line_height >= MIN_LINE_SCALE * img.shape[0]):
            #         actual_v_grouped.append(group)
            #         actual_v_points.append([
            #             group[0][0], 0, group[0][2], img.shape[0]])
            #         cv2.line(
            #             vertical_lines, (group[0][0], 0), (group[0][2], img.shape[0]), (0, 0, 0), 2)
            if(self.debug_mode_check == True):
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'image_v_re-drawn.png', vertical_lines)
            return vertical_lines, actual_h_grouped
            # return vertical_lines, actual_v_grouped
        except Exception as ex:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno
            raise Exception(str(ex)+" ((Line: "+str(lineno)+")")

    def _detect_horizontal_lines(self, img, img_bin, img_name, MAXGAP=3, COMPARE_POS=0, process_2_reqd=False):
        try:
            # image with all horizontal lines
            if(img.shape[1] <= 100):
                hor_kernel_len = 3
            elif(img.shape[1] < 150):
                hor_kernel_len = np.array(img).shape[1]//25
            elif(img.shape[1] <= 500):
                hor_kernel_len = np.array(img).shape[1]//50
            elif(img.shape[1] <= 1500):
                hor_kernel_len = np.array(img).shape[1]//100
            else:
                hor_kernel_len = np.array(img).shape[1]//200

            hor_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (hor_kernel_len, 1))
            if(process_2_reqd == True):
                image_2 = cv2.dilate(img_bin, hor_kernel, iterations=1)
                horizontal_lines = cv2.erode(image_2, hor_kernel, iterations=1)
            else:
                image_2 = cv2.erode(img_bin, hor_kernel, iterations=2)
                horizontal_lines = cv2.dilate(
                    image_2, hor_kernel, iterations=2)

            if(self.debug_mode_check == True):
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'image_h1.png', image_2)
            h_points = []
            MIN_LINE_SCALE, threshold, minLineLength, maxLineGap = 0.4, 10, img.shape[
                0]//50, 4  # img.shape[1]//80
            edges = cv2.Canny(horizontal_lines, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if(x2-x1 > 0 and y1 == y2):
                        h_points.append([x1, y1, x2, y2, x2-x1])
            # h_points.sort(key=lambda x: x[1])
            h_points = sorted(h_points, key=lambda x: (x[1], x[0]))
            # Clustering lines into one with max pixel diff of 3
            if(len(h_points) == 0):
                raise Exception(
                    "No cells detected: Horizontal Lines not detected")
            h_grouped = self._cluster(h_points, MAXGAP, 1, 2, 0, COMPARE_POS)
            horizontal_lines = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            horizontal_lines.fill(255)
            actual_h_points, actual_h_grouped = [], []
            # for each clustered line, finds the total width of line
            # and if greater than 40% of total image width it draws a complete line on the horizontal_lines image
            for group in h_grouped:
                line_height, count = 0, 0
                for line in group:
                    # if any line is greater than 10% of the image width, then consider it as potential line
                    if(line[4] > 0.05*img.shape[1]):
                        count += 1
                    line_height += line[4]
                if(line_height >= MIN_LINE_SCALE * img.shape[1] and count > 0):
                    actual_h_grouped.append(group)
                    actual_h_points.append(
                        [0, group[0][1], img.shape[1], group[0][3]])
                    cv2.line(
                        horizontal_lines, (0, group[0][1]), (img.shape[1], group[0][3]), (0, 0, 0), 2)
            if(self.debug_mode_check == True):
                cv2.imwrite(self.temp_folderpath+"\\" +
                            img_name+'image_h_re-drawn.png', horizontal_lines)
            return horizontal_lines, actual_h_grouped
        except Exception as ex:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno
            raise Exception(str(ex)+" ((Line: "+str(lineno)+")")

    def _rotate_image(self, img, angle):
        if(angle == 90):
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif(angle == 180):
            return cv2.rotate(img, cv2.ROTATE_180)
        elif(angle == 270):
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            return img

    def _deskew_image(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _array_reshape_fix(self, array, col_count, row_count):
        array = array.tolist()
        arr_size = len(array)
        if(arr_size > col_count*row_count):
            flag = True
            while(flag):
                row_count = row_count+1
                if(arr_size < col_count*row_count):
                    for i in range((col_count*row_count)-arr_size):
                        array.append(' ')
                    flag = False
                elif(arr_size == col_count*row_count):
                    break
        else:
            for i in range((col_count*row_count)-arr_size):
                array.append(' ')
        return array, col_count, row_count

    def _cluster(self, data, maxgap, position, position_1, position_2, compare_pos):
        '''Arrange data into groups where successive elements
       differ by no more than *maxgap*'''
        groups = [[data[0]]]
        for x in data[1:]:
            if abs(x[position] - groups[-1][compare_pos][position]) <= maxgap:
                if not (groups[-1][-1][position_2] <= x[position_2] <= (groups[-1][-1][position_2]+groups[-1][-1][4]) and groups[-1][-1][position_2] <= x[position_1] <= (groups[-1][-1][position_2]+groups[-1][-1][4])):
                    groups[-1].append(x)
            else:
                groups.append([x])
        return groups

    def _get_word_dict_from_hocr(self, img_arr):
        hocr_file_data = pytesseract.image_to_pdf_or_hocr(
            img_arr, extension='hocr')
        soup = bs4.BeautifulSoup(hocr_file_data, 'lxml')
        word_structure = []
        ocr_xword = soup.findAll("span", {"class": "ocrx_word"})
        for line in ocr_xword:
            line_text = line.text.replace("\n", " ").strip()
            title = line['title']
            # The coordinates of the bounding box
            conf = str(title).split(';')[1].replace('x_wconf', '').strip()
            x, y, w, h = map(int, title[5:title.find(";")].split())
            word_structure.append(
                {'text': line_text, 'bbox': [x, y, w-x, h-y], 'conf': conf})
        return word_structure

    def _second_run_extraction_processing(self, img, img_name):
        img_arr = img
        if(len(img.shape) == 2):
            img_arr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        word_structure = self._get_word_dict_from_hocr(img_arr)

        height_sum, word_count = 0, 0
        for word_dict in word_structure:
            if(word_dict['text'] != '' and re.search("^[^[|(]{1}", word_dict['text'])):
                height_sum += word_dict['bbox'][3]
                word_count += 1
                x, y, w, h = word_dict['bbox'][0], word_dict['bbox'][1], word_dict['bbox'][2], word_dict['bbox'][3]
                img_arr[y:y+h, x:x+w] = (255, 255, 255)
        cv2.imwrite(
            f'{self.temp_folderpath}\\{img_name}_removed_text.png', img_arr)
        mean_word_h = height_sum/word_count if word_count > 0 else 0
        box, warning = self._detect_all_cells(img_arr, img_name, True)
        return box, True, mean_word_h, warning


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()
    config.optionxform = str
    parser.add_argument('--ini_file', type=str,
                        help='Location of the ini file', required=True)
    args = parser.parse_args()
    config.read(args.ini_file, encoding="cp1252")
    logging_level = config['DEFAULT']['logging_level']
    log_file_path = config['DEFAULT']['log_file_path']
    now_time = datetime.datetime.now()
    timestr = now_time.strftime("%Y%m%d-%H%M%S")
    logfilename = "train_log_" + timestr + ".log"

    logging.basicConfig(filename=(log_file_path + '//' + logfilename), format='%(asctime)s- %(levelname)s- %(message)s',
                        level=eval(logging_level), datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger()

    try:
        input_path = config['DEFAULT']['path']
    except:
        self.logger.info("property path not found in config file.")

    for path in os.listdir(input_path):
        if (path.endswith('.jpg')):
            try:
                imagepath = input_path + path
                image_reader = image_table_border_reader(config, logger)
                output_dataframe = image_reader.extract(imagepath)
            except Exception as e:
                logger.info(imagepath)
    help(image_table_border_reader)
    help(image_table_border_reader.extract)