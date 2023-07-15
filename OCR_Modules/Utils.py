from PyPDF2 import PdfFileWriter, PdfFileReader
from project.server.main.Modules.DocumentMiner.FormTypeDetection.unicode_correction import  Englishify
from project.server.main.Modules.DocumentReader.AzureOCR import AzureOcr
from project.server.main.Modules.DocumentReader.PDFMiner import PdfMiner
from project.server.main.Modules.DocumentMiner.FormTypeDetection.FormClassify import FormTypeDetection,PageTypeDetection

import os
PAGE_LIMIT =1999
SPLIT_PAGE_LIMIT=4
def get_all_text_from_azure(pdf_path,temp_dir,page_params=None):
    try:
        pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages
        print("split_path",pdf_path,total_pages)
        if total_pages>PAGE_LIMIT:
            #If number of pages is greater than 2000, we take only first 2000 pages
            page_params={'pages': "1-2000"}

        # Text Extraction Using AzureOCR
        print("page_params",page_params)
        azure_ocr_obj = AzureOcr(pdf_path,page_params=page_params)
        is_scanneds,dataframes_tuple,isREADAPIfailed = azure_ocr_obj.main()
        if not isREADAPIfailed:
            azure_read_pages = len(set(dataframes_tuple.df["page"].unique()))
            save_ocr_results(pdf_path,temp_dir,dataframes_tuple,tag='1')
        else:
            print("Azure READ API Failed")
            is_scanneds=None
            dataframes_tuple=False
            azure_read_pages=0


    except Exception as e:
        print("Code Failed while running get_all_text_from_azure with error ",str(e))
        is_scanneds=None
        dataframes_tuple=False
        azure_read_pages=0
        isREADAPIfailed = True
    return is_scanneds,dataframes_tuple,azure_read_pages,isREADAPIfailed


def get_form_type_with_start_page_AzureOCR(pdf_path,temp_dir):
         
        split_path, total_pages = get_split_pdf(pdf_path,temp_dir)
        print("split_path",split_path,total_pages)
        if not split_path:
            split_path = pdf_path
        pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages
        azure_ocr_obj = AzureOcr(split_path,timeout=15)
        is_scanned,dataframes_tuple,isREADAPIfailed = azure_ocr_obj.main()
        if not isREADAPIfailed:
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
            page_dict_info={}
            if form_type in ['dd1423','dd1155']:
                page_type_obj = PageTypeDetection(df_page_text,form_type,start_page)
                page_dict_info = page_type_obj.main()
            variable_dict={}
            variable_dict['form_type'] = form_type
            variable_dict['start_page'] = start_page
            variable_dict['form_sub'] = form_sub
            variable_dict['scannedpdf'] = is_scanned
            variable_dict['canusepdfminer'] = False
            variable_dict['split_path'] = split_path
            variable_dict['total_pages'] = total_pages
            variable_dict['page_dict_info']=page_dict_info
            variable_dict['read_api_failed_status']=isREADAPIfailed
            variable_dict['Form_type_With_sub'] = str(form_type) +'-'+ str(form_sub) if form_sub else str(form_type)
        else:
            print("READ API Failed")
            variable_dict={}
            variable_dict['form_type'] = None
            variable_dict['read_api_failed_status']=True
            dataframes_tuple=False
        # except Exception as e:
        #     print("Code Failed while running get_form_type_with_start_page with error ",str(e))
        #     variable_dict={}
        #     variable_dict['form_type'] = None
        #     dataframes_tuple=False
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




def get_split_pdf(pdf_path,temp_path):
    """
            *Description: Split the first 3 pages of the pdf, which is used to check the Form-Type
            *Parameters: pdf_path - path to pdf
            *Returns: split_pdf_path(False in case not able to split), total_pages 
    """
    #pages ={'pages':'1-3'}
    # #split_pdf_path = pathlib.PosixPath(constants.pdfs_path, 'splitpages.pdf')
    split_pdf_path = os.path.join(temp_path, 'splitpages_first4.pdf')
    #print(constants.pdfs_path,split_pdf_path)
    # if not os.path.exists(constants.pdfs_path):
    #     os.mkdir(constants.pdfs_path)
    # pdf_reader = PdfFileReader(open(pdf_path, "rb"))
    pdf_reader = PdfFileReader(open(pdf_path, "rb"),strict=False)
    total_pages = pdf_reader.numPages
    #print(pdf_reader.numPages,"Number of pages")
    if total_pages >= SPLIT_PAGE_LIMIT:
        try:
            pdf_writer1 = PdfFileWriter()
            for page in range(SPLIT_PAGE_LIMIT):
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

