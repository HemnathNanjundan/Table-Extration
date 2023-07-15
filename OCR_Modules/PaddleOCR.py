from pdf2image import convert_from_path
from PIL import Image
from paddleocr import PaddleOCR
import pandas as pd
import json
import os
import tempfile
import time
import copy
from datetime import timedelta
# from Modules.textextractor.azure_ocr import line_numbers,genearte_dataframes
from  project.server.main.Modules.DocumentReader.AzureOCR import line_numbers,generate_dataframes

class PaddleOcr:
    PP_OCR = PaddleOCR(use_gpu= False )
    def __init__(self,file_path):
        self.file_path = file_path

    def __processOcr(self,img_path,img_id):
        ocr_result = self.PP_OCR.ocr(img_path, cls= False )
        image = Image.open(img_path).convert('RGB')
        wid,hei = image.size
        boxes=[]
        scores=[]
        page_line_by_line=[]
        txts=[]
        page,angle,width,height,unit= img_id,1,wid,hei,'inch'
        for each_line in ocr_result[0]:
            # bb_boxes = each_line[0]
            boxes.append(each_line[0])
            scores.append(each_line[1][1])
            Description = each_line[1][0]
            txts.append(Description)
            top_left,top_right,bottom_right, bottom_left = each_line[0]
            Top_left_x, Top_left_y = top_left
            Top_right_x, Top_right_y = top_right
            Bottom_right_x, Bottom_right_y = bottom_right
            Bottom_left_x, Bottom_left_y = bottom_left

            page_line_by_line.append([Description, self.__inches(Top_left_x), self.__inches(Top_left_y), self.__inches(Top_right_x), self.__inches(Top_right_y), self.__inches(Bottom_right_x), self.__inches(Bottom_right_y), self.__inches(Bottom_left_x), self.__inches(Bottom_left_y) ] + [page, angle, width, height, unit])
        return page_line_by_line

    def __get_text_from_ppocr(self):
        print("input_filename",self.file_path)
        details_line_by_line = []
        with tempfile.TemporaryDirectory() as path:
            images_from_path = convert_from_path(self.file_path,dpi=300, output_folder=path,fmt="jpeg",paths_only=True)
            for pg, each_image in enumerate(images_from_path):
                print(pg,each_image)
                s_time = time.time()
                image_id = pg+ 1
                page_deatils = self.__processOcr(each_image,image_id)
                details_line_by_line.extend(page_deatils)
                
                e_time = time.time()
                print('Prediction time for page {} is {}:'.format(pg, str(timedelta(seconds=e_time-s_time))))
        df_line_ppocr = pd.DataFrame(details_line_by_line, columns = ['Description', 
                                'Top_left_x','Top_left_y' ,
                                'Top_right_x', 'Top_right_y' ,
                                'Bottom_right_x','Bottom_right_y',
                                'Bottom_left_x','Bottom_left_y',
                                "page", "angle", "width", "height", "unit"])
        df_line_ppocr, is_scanned = line_numbers(df_line_ppocr)
        df_ppocr = copy.deepcopy(df_line_ppocr)
        dataframes_tuple = generate_dataframes(df_ppocr,df_line_ppocr,is_scanned,self.file_path)

        return is_scanned,dataframes_tuple


    def __inches(self,i):
        d=i/300
        e=round(d,2)
        return e

    def main(self):
        try:
            is_scanneds,dataframes_tuplse =PaddleOcr.__get_text_from_ppocr(self)
        except Exception as e:
            return e
        return is_scanneds,dataframes_tuplse
    
    # def main(self):
    #     is_scanneds,dataframes_tuplse =PaddleOcr.__get_text_from_ppocr(self)
    #     return is_scanneds,dataframes_tuplse


if __name__ == "__main__":
    paddle_ocr_obj = PaddleOcr('/Users/shyamz/TM/revamp/tm-ici-app/temp/am624/splitpages_first3.pdf')
    is_scanneds,dataframes_tuplse = paddle_ocr_obj.main()
    print(is_scanneds,dataframes_tuplse)

