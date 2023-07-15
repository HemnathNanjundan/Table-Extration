from PyPDF2 import PdfFileWriter, PdfFileReader
from project.server.main.Modules.DocumentMiner.FormTypeDetection.unicode_correction import  Englishify
from project.server.main.Modules.DocumentReader.AzureOCR import AzureOcr
from project.server.main.Modules.DocumentReader.PaddleOCR import PaddleOcr
from project.server.main.Modules.DocumentReader.PDFMiner import PdfMiner
from project.server.main.Modules.DocumentMiner.FormTypeDetection.FormClassify import FormTypeDetection

import os

def get_form_type_with_start_page_old(pdf_path,temp_dir):
    try:

        split_path, total_pages = get_split_pdf(pdf_path,temp_dir)
        print("split_path",split_path,total_pages)
        if not split_path:
            split_path = pdf_path
        pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages

        # Text Extraction Using PDFMiner
        pminer_ocr_obj = PdfMiner(split_path)
        is_scanned,use_pdfminer,dataframes_tuple = pminer_ocr_obj.main()

        if not use_pdfminer:
            # Text Extraction Using PaddleOcr
            # dataframes_tuple = azureOCR(split_path)
            print("SCANED PDF, USING PPOCR")
            # is_scanned,dataframes_tuple = get_text_from_ppocr(split_path)
            try:
                paddle_ocr_obj = PaddleOcr(split_path)
                is_scanned,dataframes_tuple = paddle_ocr_obj.main()
            except:
                #Using Azure Code
                azure_ocr_obj = AzureOcr(split_path)
                is_scanneds,dataframes_tuple = azure_ocr_obj.main()
            use_pdfminer = False
        df_page_text = dataframes_tuple.df_text
        form_det_obj = FormTypeDetection(df_page_text)
        form_type , start_page,form_sub, pagestosplit = form_det_obj.main()    
        if form_type in ['of-347'] and pagestosplit:
            if pagestosplit < total_pages:
                split_path = get_split_pdf_by_pagenumber(pdf_path,temp_dir,pagestosplit)
            elif pagestosplit == total_pages:
                split_path = pdf_path
        elif form_type in ['of-347'] and not pagestosplit:
            split_path = pdf_path
        variable_dict={}
        variable_dict['form_type_value'] = form_type
        variable_dict['start_page_value'] = start_page
        variable_dict['form_specific_value'] = form_sub
        # variable_dict['dataframe'] = dataframes_tuple
        variable_dict['is_scanned'] = is_scanned
        variable_dict['use_pdfminer'] = use_pdfminer
        variable_dict['split_path'] = split_path
        variable_dict['v_total_pages'] = total_pages
        save_ocr_results(pdf_path,temp_dir,dataframes_tuple,tag='F3')

    except Exception as e:
        print("Code Failed while running get_form_type_with_start_page with error ",str(e))
        variable_dict={}
        dataframes_tuple=False
    return variable_dict,dataframes_tuple


def get_form_type_with_start_page(pdf_path,temp_dir):
    try:

        split_path, total_pages = get_split_pdf(pdf_path,temp_dir)
        print("split_path",split_path,total_pages)
        if not split_path:
            split_path = pdf_path
        pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages

        # Text Extraction Using PDFMiner

        ispdfminer_success,form_type,start_page,form_sub,pagestosplit,dataframes_tuple,use_pdfminer= get_form_type_with_start_page_PDFMiner(pdf_path,temp_dir)
        
        if not ispdfminer_success: #IF PDFMINER FAILED

            paddle_ocr_obj = PaddleOcr(split_path)
            is_scanned,dataframes_tuple = paddle_ocr_obj.main()
            # use_pdfminer = False
            if dataframes_tuple:

                df_page_text = dataframes_tuple.df_text
                form_det_obj = FormTypeDetection(df_page_text)
                form_type , start_page,form_sub, pagestosplit = form_det_obj.main() 
            else:
                variable_dict={}
                variable_dict['form_type'] = None
                return variable_dict,dataframes_tuple

        if not use_pdfminer:
            is_scanned=True
        else:
            is_scanned=False
        
        if form_type in ['of-347'] and pagestosplit:
            if pagestosplit < total_pages:
                split_path = get_split_pdf_by_pagenumber(pdf_path,temp_dir,pagestosplit)
            elif pagestosplit == total_pages:
                split_path = pdf_path
        elif form_type in ['of-347'] and not pagestosplit:
            split_path = pdf_path
        variable_dict={}
        variable_dict['form_type'] = form_type
        variable_dict['start_page'] = start_page
        variable_dict['form_sub'] = form_sub
        variable_dict['Form_type_With_sub'] = str(form_type) + str(form_sub) if form_sub else str(form_type)
        variable_dict['scannedpdf'] = is_scanned
        variable_dict['canusepdfminer'] = use_pdfminer
        variable_dict['split_path'] = split_path
        variable_dict['total_pages'] = total_pages
        # save_ocr_results(pdf_path,temp_dir,dataframes_tuple,tag='F3')

    except Exception as e:
        print("Code Failed while running get_form_type_with_start_page with error ",str(e))
        variable_dict={}
        variable_dict['form_type'] = None
        dataframes_tuple={}
    return variable_dict,dataframes_tuple



def get_form_type_with_start_page_PaddleOCR(pdf_path,temp_dir):
    try:

        split_path, total_pages = get_split_pdf(pdf_path,temp_dir)
        print("split_path",split_path,total_pages)
        if not split_path:
            split_path = pdf_path
        pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages
        paddle_ocr_obj = PaddleOcr(split_path)
        is_scanned,dataframes_tuple = paddle_ocr_obj.main()

        df_page_text = dataframes_tuple.df_text
        form_det_obj = FormTypeDetection(df_page_text)
        form_type , start_page,form_sub, pagestosplit = form_det_obj.main()    
        if form_type in ['of-347'] and pagestosplit:
            if pagestosplit < total_pages:
                split_path = get_split_pdf_by_pagenumber(pdf_path,temp_dir,pagestosplit)
            elif pagestosplit == total_pages:
                split_path = pdf_path
        elif form_type in ['of-347'] and not pagestosplit:
            split_path = pdf_path
        variable_dict={}
        variable_dict['form_type_value'] = form_type
        variable_dict['start_page_value'] = start_page
        variable_dict['form_specific_value'] = form_sub
        # variable_dict['dataframe'] = dataframes_tuple
        variable_dict['is_scanned'] = is_scanned
        variable_dict['use_pdfminer'] = False
        variable_dict['split_path'] = split_path
        variable_dict['v_total_pages'] = total_pages
    except Exception as e:
        print("Code Failed while running get_form_type_with_start_page with error ",str(e))
        variable_dict={}
        dataframes_tuple=False
    return variable_dict,dataframes_tuple



def get_form_type_with_start_page_PDFMiner(pdf_path,temp_dir):
    ispdfminer_success,form_type ,start_page,form_sub,pagestosplit = False,False,False,False,False
    try:
        # ispdfminer_success,form_type ,start_page,form_sub,pagestosplit = False,False,False,False,False
        split_path, total_pages = get_split_pdf(pdf_path,temp_dir)
        print("split_path",split_path,total_pages)
        if not split_path:
            split_path = pdf_path
        pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages
        
        pminer_ocr_obj = PdfMiner(split_path)
        is_scanned,use_pdfminer,dataframes_tuple = pminer_ocr_obj.main()
        if  use_pdfminer:
            form_det_obj = FormTypeDetection(dataframes_tuple.df_text)
            form_type , start_page,form_sub, pagestosplit = form_det_obj.main()  
            if form_type:
                ispdfminer_success = True
        # return  ispdfminer_success,form_type,start_page,form_sub,pagestosplit,dataframes_tuple,use_pdfminer

    except Exception as e:
        print("Code Failed while running get_form_type_with_start_page with error ",str(e))
        dataframes_tuple=False
    return ispdfminer_success,form_type,start_page,form_sub,pagestosplit,dataframes_tuple,use_pdfminer


def get_split_pdf(pdf_path,temp_path):
    """
            *Description: Split the first 3 pages of the pdf, which is used to check the Form-Type
            *Parameters: pdf_path - path to pdf
            *Returns: split_pdf_path(False in case not able to split), total_pages 
    """
    #pages ={'pages':'1-3'}
    # #split_pdf_path = pathlib.PosixPath(constants.pdfs_path, 'splitpages.pdf')
    split_pdf_path = os.path.join(temp_path, 'splitpages_first3.pdf')
    #print(constants.pdfs_path,split_pdf_path)
    # if not os.path.exists(constants.pdfs_path):
    #     os.mkdir(constants.pdfs_path)
    # pdf_reader = PdfFileReader(open(pdf_path, "rb"))
    pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
    total_pages = pdf_reader.numPages
    #print(pdf_reader.numPages,"Number of pages")
    if total_pages >= 3:
        try:
            pdf_writer1 = PdfFileWriter()
            for page in range(3):
                pdf_writer1.addPage(pdf_reader.getPage(page))            
            with open(split_pdf_path, 'wb') as file2:
                pdf_writer1.write(file2)
            return split_pdf_path,total_pages
        except:
            print("Unable to split the form")
            return False,total_pages
    else:
        return False,total_pages

def get_split_pdf_by_pagenumber(pdf_path,temp_dir,page_number):

    #split_pdf_path = pathlib.PosixPath(constants.pdfs_path, 'splitpages.pdf')
    split_pdf_path = os.path.join(temp_dir, 'splitpages.pdf')

    # pdf_reader = PdfFileReader(open(pdf_path, "rb"))
    pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
    #print(pdf_reader.numPages,"Number of pages")

    if pdf_reader.numPages >= page_number:
        try:
            pdf_writer1 = PdfFileWriter()
            for page in range(page_number):
                pdf_writer1.addPage(pdf_reader.getPage(page))
            
            with open(split_pdf_path, 'wb') as file2:
                pdf_writer1.write(file2)
            return split_pdf_path
        except:
            print("Unable to split pages from get_split_pdf_by_pagenumber")
            return pdf_path
    else:
        return pdf_path


def save_ocr_results(file_name,save_path,azureDF,tag):
    # azureDF,json_file = azureOCR(file_name,page_params=None)
    azure_df_text = azureDF.df_text
    azure_df_text_header_footer_removal = azureDF.df_text_header_footer_removal
    azure_df_line = azureDF.df_line
    azure_df = azureDF.df
    head, tail = os.path.split(file_name)
    file_sname = tail.split('.')[0] + "{}_df_text.csv".format(tag)
    file_s1name = tail.split('.')[0] + "{}_df_text_header_footer_removal.csv".format(tag)
    file_s2name = tail.split('.')[0] + "{}_df.csv".format(tag)
    file_s3name = tail.split('.')[0] + "{}_df_line.csv".format(tag)
    
    save_p1 = os.path.join(save_path,file_sname)
    savep2 = os.path.join(save_path,file_s1name)
    save_p3 = os.path.join(save_path,file_s2name)
    savep3 = os.path.join(save_path,file_s3name)

    azure_df_text.to_csv(save_p1)
    azure_df_text_header_footer_removal.to_csv(savep2)
    azure_df.to_csv(save_p3)
    azure_df_line.to_csv(savep3)
    return "success"
