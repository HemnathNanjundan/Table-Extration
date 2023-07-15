import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import statistics
from pdfminer.layout import LAParams, LTTextBox,LTPage,LTText
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import copy

# lAparams = LAParams(detect_vertical= True, char_margin= 1.5,line_margin= .1,all_texts= True)

# from Modules.textextractor.azure_ocr import line_numbers,genearte_dataframes
from  project.server.main.Modules.DocumentReader.AzureOCR import line_numbers,generate_dataframes

class PdfMiner:
    def __init__(self,file_path):
        self.file_path = file_path

    def __inches(self,i,dpi_value=1):
        d=i*(dpi_value/72)
        e=round(d,2)
        return e


    def __pdfminer_parsing(self):
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(detect_vertical= True, char_margin= 1.5,line_margin= .1,all_texts= True)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        fp = open(self.file_path, 'rb')
        pages = PDFPage.get_pages(fp)    
        details_line_by_line = []
        vertical_shift=0
        # dpi = 200/72
        page_cnt={}
        for page_id, page in enumerate(pages):
            interpreter.process_page(page)
            layout = device.get_result()
            page_width = layout.width
            page_heigt = layout.height
            page_angle=layout.rotate
            print(page_width,page_heigt)
            page,angle,width,height,unit= page_id+1,page_angle,page_width,page_heigt,'inch'
            count=0
            for lobj in layout._objs:
                # print(type(lobj),"TYPE",count)
                # if isinstance(lobj, LTTextBox):
                if isinstance(lobj,LTText):
                    
                    x1,y1,x2,y2, Description = lobj.bbox[0], lobj.bbox[1],lobj.bbox[2],lobj.bbox[3], lobj.get_text()
                    startX = x1
                    startY = page_heigt - y1 - vertical_shift
                    endX   = x2
                    endY   = page_heigt - y2 - vertical_shift
                    top_left,top_right,bottom_right, bottom_left = ([x1,endY],[x2,endY],[x2,startY],[x1,startY])

                    Top_left_x, Top_left_y = top_left
                    Top_right_x, Top_right_y = top_right
                    Bottom_right_x, Bottom_right_y = bottom_right
                    Bottom_left_x, Bottom_left_y = bottom_left
                    details_line_by_line.append([Description.strip(), self.__inches(Top_left_x), self.__inches(Top_left_y), self.__inches(Top_right_x), self.__inches(Top_right_y), self.__inches(Bottom_right_x), self.__inches(Bottom_right_y), self.__inches(Bottom_left_x), self.__inches(Bottom_left_y) ] + [page, angle, width, height, unit])
                    # count = count + 1
                    count +=1
                else:
                    # print(type(lobj))
                    pass
                    # print("Not finmd any text here",dir(lobj))
            page_cnt[page_id]=count

        ms = statistics.mode(list(page_cnt.values()))
        print(ms,"MS Value")
        if ms >=15:
            df_line_miner = pd.DataFrame(details_line_by_line, columns = ['Description', 
                                'Top_left_x','Top_left_y' ,
                                'Top_right_x', 'Top_right_y' ,
                                'Bottom_right_x','Bottom_right_y',
                                'Bottom_left_x','Bottom_left_y',
                                "page", "angle", "width", "height", "unit"])
            df_line_miner, is_scanned = line_numbers(df_line_miner)
            df_pdfminer = copy.deepcopy(df_line_miner)

            dataframes_tuple = generate_dataframes(df_pdfminer,df_line_miner,is_scanned,self.file_path)
            use_pdfminer = True
        else:
            use_pdfminer = False
            dataframes_tuple ={}
            is_scanned = True

            print("Need to perform OCR")
    
        return is_scanned,use_pdfminer,dataframes_tuple

    def main(self):
        try:
            is_scanneds,use_pdfminer,dataframes_tuplse =PdfMiner.__pdfminer_parsing(self)
        except Exception as e:
            return e
        return is_scanneds,use_pdfminer,dataframes_tuplse


if __name__ == "__main__":
    pminer_ocr_obj = PdfMiner('/Users/shyamz/TM/revamp/tm-ici-app/temp/am624/splitpages_first3.pdf')
    is_scanneds,use_pdfminer,dataframes_tuplse = pminer_ocr_obj.main()
    print(is_scanneds,use_pdfminer,dataframes_tuplse)