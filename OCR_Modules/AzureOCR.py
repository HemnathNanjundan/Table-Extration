from collections import namedtuple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import requests
import time
import re
import copy
from PyPDF2 import PdfFileWriter, PdfFileReader
from sklearn.cluster import MeanShift, estimate_bandwidth
from project.server.main.Modules.DocumentMiner.FormTypeDetection import data_config
from project.server.main.Modules.DocumentMiner.FormTypeDetection.FormClassify import generate_classifer_dict

from project.server.main.Modules.settings import localconf as BaseConfig

# from project.server.config import BaseConfig
subscription_key = BaseConfig.READ_API_SUBSCRIPTION_KEY
endpoint = BaseConfig.READ_API_ENDPOINT

# text_recognition_url = endpoint + "/vision/v3.2/read/analyze"
# text_recognition_url = endpoint + "/vision/v3.1/read/syncAnalyze"

CHECK_N_PAGES = 7
PAGE_SPLIT =  None
NUM_LINE_HEADER = 5
NUM_LINE_FOOTER = 5
HEADER_FOOTER_THRESHOLD = 10
section_pattern_dic = generate_classifer_dict(data_config.section_type_keys)



def line_numbers(df:pd.DataFrame, by_pixel_value:int = 16.6) -> pd.DataFrame:
    """
    *Author: Vaibhav
    *Details: Logic for getting accurate line number irrespective of any noise 
              in word coordinates coming from hocr file.     
              Input: DataFrame
              Output: Updated DataFrame with a new constants.MISC_LINE_NUMBER column in it.
    *@param: DataFrame, float
    *@return: DataFrame
    """
    is_scanned = True
    for index in range(1,9):
        df_type = df.iloc[:,index].dtype
        if df_type == float:
            continue
        else:
            break
    else:
        is_scanned = False

    if is_scanned:
        by_pixel_value = by_pixel_value
    else:
        by_pixel_value = 0.048

    merged_dataframe = pd.DataFrame()
    for page in sorted(set(df['page'])):
        df_per_page = df[df['page'] == page]
        df_per_page = df_per_page.sort_values(by=['Top_left_y', 'Top_left_x'], ascending = [True, True]).reset_index(drop = True)    
        # y_avg = df_per_page.loc[: , [constants.MISC_LEFT_Y,constants.MISC_RIGHT_Y]]
        y_avg = df_per_page.loc[: , ["Top_left_y"]]
        data = pd.DataFrame()
        data['Top_y_avg'] = y_avg.mean(axis=1)
        data['Top_y_difference'] = data['Top_y_avg'].diff()
        data['IS_PIXEL'] = data['Top_y_difference'] > by_pixel_value
        df_copy = data[:]
        df_copy = df_copy.reset_index(drop = True)
        line_number = []
        class_number, loop = 0, 0
        while(loop < len(df_per_page)):
            if(df_copy['IS_PIXEL'][loop] == False):
                line_number.append(class_number)
            elif(df_copy['IS_PIXEL'][loop] == True):
                class_number+=1
                line_number.append(class_number)
            loop+=1
 
        previously_total_lines = not merged_dataframe.empty and max(set(merged_dataframe['line_number']))
        df_per_page['line_number'] = [line + previously_total_lines+1 for line in line_number]
            
        df_per_page = df_per_page.sort_values(by=['line_number', 'Top_left_x', 'Bottom_right_y'], ascending = [True, True, True]).reset_index(drop = True)
        # merged_dataframe = merged_dataframe.append(df_per_page)
        merged_dataframe =pd.concat([merged_dataframe, df_per_page],axis=0,ignore_index=True)

    
    return merged_dataframe, is_scanned


def calculate_paragraph_and_column(df_line):
    df_line['vertical_text_lines'] = None
    df_line['paragraph'] = None
    df_line['column'] = None
    for page_number in sorted(set(df_line['page'])):
        # print("page_number: {}".format(page_number))
        df_line = calculating_paragraph(df_line, page_number)

    # Calculating paragraph_number column for complete PDF
    paragraph_number = []
    count = 0
    prev_para_num = df_line['paragraph'].tolist() and df_line['paragraph'].tolist()[0]
    for para_num in df_line['paragraph']:
        if para_num==prev_para_num or pd.isna(para_num):
            pass
        else:
            count += 1
            prev_para_num = para_num        
        paragraph_number.append(count)
    df_line['paragraph_number'] = paragraph_number
    return df_line


def calculating_paragraph(df_line, page_number):
    """
    Creating paragraph attribute for calculating paragraph number of the text
    present in given dataframe using clustering on coordiantes.
    
    Input : 
        DataFrame
        Page Number to calculates paragraphs on that page
    """
    MIN_LINE_SPACE = 0.09

    df_line = df_line.reset_index(drop=True)    

    # Operation on page
    page_df = df_line[df_line['page']==page_number]

    # Calculating vertical text
    page_df = page_df.copy()
    page_df['x_diff'] = page_df['Top_right_x']-page_df['Top_left_x']
    page_df['y_diff'] = page_df['Top_right_y']-page_df['Top_left_y']
    temp_page_df = page_df[page_df['x_diff']==0]    
    v_df = pd.DataFrame(index=temp_page_df['Top_left_x'], columns=['Description', 'line_number'])
    v_df['Description'] = temp_page_df['Description'].tolist()
    v_df['line_number'] = temp_page_df['line_number'].tolist()

    my_line_num_text_dict = v_df.T.to_dict()
    
    page_df.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]
    df_line.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]
    
    
    dd = pd.DataFrame(index = temp_page_df.index)
    dd['Top_left_x'] = temp_page_df['Top_right_x'].tolist()
    dd['Top_left_y'] = temp_page_df['Top_right_y'].tolist()
    
    dd['Top_right_x'] = temp_page_df['Bottom_right_x'].tolist()
    dd['Top_right_y'] = temp_page_df['Bottom_right_y'].tolist()
    
    dd['Bottom_right_x'] = temp_page_df['Bottom_left_x'].tolist()
    dd['Bottom_right_y'] = temp_page_df['Bottom_left_y'].tolist()
    
    dd['Bottom_left_x'] = temp_page_df['Top_left_x'].tolist()
    dd['Bottom_left_y'] = temp_page_df['Top_left_y'].tolist()

    if not dd.empty:
        dd['Top_left_x'] = min(dd['Top_left_x'])

    page_df.loc[dd.index, ['Top_left_x', 'Top_left_y', 'Top_right_x', 'Top_right_y',
       'Bottom_right_x', 'Bottom_right_y', 'Bottom_left_x', 'Bottom_left_y']] = dd.loc[dd.index, ['Top_left_x', 'Top_left_y', 'Top_right_x', 'Top_right_y',
       'Bottom_right_x', 'Bottom_right_y', 'Bottom_left_x', 'Bottom_left_y']]
                                                                                                              
    df_line.loc[dd.index, ['Top_left_x', 'Top_left_y', 'Top_right_x', 'Top_right_y',
       'Bottom_right_x', 'Bottom_right_y', 'Bottom_left_x', 'Bottom_left_y']] = dd.loc[dd.index, ['Top_left_x', 'Top_left_y', 'Top_right_x', 'Top_right_y',
       'Bottom_right_x', 'Bottom_right_y', 'Bottom_left_x', 'Bottom_left_y']]
    
    # page_df = page_df[pd.isna(page_df['vertical_text_lines'])]
    

    # Assigning approprate value for coordinated belonging to same line
    for li in sorted(set(page_df.line_number)):
        df_li = page_df[page_df['line_number']==li]
        page_df = page_df.copy()
        page_df.loc[df_li.index, 'Bottom_right_y'] = max(df_li['Bottom_right_y'])
        page_df.loc[df_li.index, 'Top_left_y'] = min(df_li['Top_left_y'])
        page_df.loc[df_li.index, 'Bottom_left_y'] = max(df_li['Bottom_left_y'])
        page_df.loc[df_li.index, 'Top_right_y'] = min(df_li['Top_right_y'])
        
    # Calculating y-coordinates space above and below line
    page_df['bottom'] = [0] + page_df['Bottom_right_y'].tolist()[:-1]
    page_df['up_space'] = page_df['Top_left_y'] - page_df['bottom']
    page_df['down_space'] = page_df['up_space'][1:].tolist()+ [0]
    
    # Assigning approprate value for coordinated belonging to same line
    for li in sorted(set(page_df.line_number)):
        df_li = page_df[page_df['line_number']==li]
        page_df.loc[df_li.index, 'up_space'] = max(df_li['up_space'])
        page_df.loc[df_li.index, 'down_space'] = max(df_li['down_space'])
        
    # Filter for eliminating large bottom blank space before clustering
    page_df1 = page_df[page_df['up_space'] < 1.8]
    page_df2 = page_df[page_df['up_space'] >= 1.8]
    
    if page_df1.empty:
        return df_line
    
    # MeanShift Clustering in space between two lines
    X = np.array(page_df1.loc[:, ['up_space']])
    model1 = MeanShift()

    # fit model and predict clusters
    yhat = model1.fit_predict(X)

    # Adding -1 cluster number for ignored words below large bottom blank space
    page_df['yhat'] = list(yhat) + [-1 for _ in range(len(page_df2))]
    
    # Sorting clustering number bases on upper space of line
    page_df = page_df.sort_values(by=['up_space'])

    # Reordering clustering in ascending order based on height of upper blank space of line
    yhat_ascending_sequence = []
    count = 0
    prev_cluster_no = page_df['yhat'].tolist() and page_df['yhat'].tolist()[0]
    for cluster_no in page_df['yhat']:
        if prev_cluster_no != cluster_no:
            count += 1
        yhat_ascending_sequence.append(count)
        prev_cluster_no = cluster_no
    
    page_df['yhat'] = yhat_ascending_sequence
    page_df = page_df.sort_index()
    
    # Creating paragraph sequence by combining 0 with non-zerp values and lines whose upper space is less than MIN_LINE_SPACE
    paragraph_seq = []
    count = 0
    prev_line = page_df['line_number'].tolist() and page_df['line_number'].tolist()[0]
    for y, line, up_space in zip(page_df['yhat'], page_df['line_number'], page_df['up_space']):
        if y and line != prev_line:
            if up_space > MIN_LINE_SPACE:
                count += 1
        prev_line = line
        paragraph_seq.append(count)
    
    # Adding paragraph number and sorting results
    page_df['paragraph'] = paragraph_seq
    page_df= page_df.sort_values(by=['line_number', "Top_left_x"])

    # MeanShift Clustering in top left x coordinates
    X = np.array(page_df.loc[:, ['Top_left_x']])
    bandwidth = estimate_bandwidth(X, quantile=0.16, n_samples=500)
    if bandwidth:
        model2 = MeanShift(bandwidth=bandwidth)
    else:
        model2 = MeanShift()
    xhat = model2.fit_predict(X)
    cluster_centers = model2.cluster_centers_
    page_df['xhat'] = xhat 
    
    # Sorting clustering number bases on Top left x of line
    page_df = page_df.sort_values(by=['Top_left_x'])
    
    # Reordering clustering in ascending order based on height of upper blank space of line
    xhat_ascending_sequence = []
    count = 0
    prev_cluster_no = page_df['xhat'].tolist() and page_df['xhat'].tolist()[0]
    for cluster_no in page_df['xhat']:
        if prev_cluster_no != cluster_no:
            count += 1
        xhat_ascending_sequence.append(count)
        prev_cluster_no = cluster_no
    
    page_df['column'] = xhat_ascending_sequence
    page_df = page_df.sort_index()
    
    # Assignment of value to df_line
    df_line.loc[page_df.index, 'up_space'] = page_df['up_space']
    df_line.loc[page_df.index, 'down_space'] = page_df['down_space']
    df_line.loc[page_df.index, 'xhat'] = page_df['xhat']
    df_line.loc[page_df.index, 'yhat'] = page_df['yhat']
    df_line.loc[page_df.index, 'paragraph'] = page_df['paragraph']
    df_line.loc[page_df.index, 'column'] = page_df['column']
    return df_line




def hashify(sentence_line):
    hash_boolean = [x=='' for x in sentence_line] + \
                   [x.isnumeric() * 1 for x in sentence_line] + \
                   [x.isalpha() * 100 for x in sentence_line] + \
                   [x.islower() * 500 for x in sentence_line] + \
                   [x.isupper() * 300 for x in sentence_line] + \
                   [len(x) for x in sentence_line.split()]
    
    hash_val = sum(hash_boolean)
    return hash_val    

#  dataframes_tuple = generate_dataframes(df_azure,df_line_azure,is_scanned,)

def generate_dataframes(df,df_line,is_scanned,file_path,ocr_response=None,total_time_of_post_request=0,total_time_of_get_request=0,ocr_parse_time=0):
    print("here in generate")
    OCR = namedtuple('OCR', '''file_name, 
                    ocr_response
                     df
                     df_text
                     df_text_header_footer_removal
                     is_scanned
                     df_text_two_column_support
                     df_text_two_column_support_header_footer_removal
                     df_line    
                     header_dict
                     footer_dict
                     PAGE_SPLIT
                     is_hierarchy
                     total_time_of_post_request
                     total_time_of_get_request
                     ocr_parse_time
                     header_footer_removing_time
                     determing_and_splitting_two_column_document_time 
                     ''')
    
    tic = time.time()
    space_list = []
    distance_list = []
    line_sentence_list = []
    line_page_list = []
    lines = []
    ascii_ord = []
    bottom_left_x = []
    bottom_left_y = []
    for every_line_number in sorted(set(df['line_number'])):
        df_data = df[df['line_number'] == every_line_number]
        space = []
        for left, right in zip(df_data['Top_left_x'][1:], df_data['Top_right_x'][:-1]):
            space.append(left-right)
        else:
            space_list.extend(space)
            space_list.append(12)
            # Avg space is 12 pixel
            distance = []
            first_word_coordinated_in_line = df_data['Top_left_x'].loc[df_data.index[0]]
            distance.append(round(first_word_coordinated_in_line/12))
            for i in space:
                if i/12:
                    distance.append(i/12)
                else:
                    distance.append(1)
            # else:
            #     distance.append(1)
            distance_list.extend(distance)
        line_sentence = ''
        for index, word in enumerate(df_data['Description'].tolist()):
            if int(round(distance[index])):
                space_distance = ' '* int(round(distance[index]))
                line_sentence += space_distance + word
            else:
                line_sentence += ' ' + word
        else:
            line_sentence_list.append(line_sentence)            
            ascii_val = [ord(letter) for letter in line_sentence]
            ascii_ord.append(ascii_val)
            if list(df_data['page'].unique()):
                line_page_list.append(sorted(df_data['page'].unique())[0])
            else:
                line_page_list.append(None)
                
            if list(df_data['Bottom_left_x'].unique()):
                bottom_left_x.append(sorted(df_data['Bottom_left_x'].unique())[0])
            else:
                bottom_left_x.append(None)
            if list(df_data['Bottom_left_y'].unique()):
                bottom_left_y.append(sorted(df_data['Bottom_left_y'].unique())[0])
            else:
                bottom_left_y.append(None)
            lines.append(every_line_number)
    # print("Before sec_name_list",len(line_sentence_list))
    # sec_name_list = get_page_sections_from_list(line_sentence_list)
    df_text = pd.DataFrame()
    df_text['sentences'] = line_sentence_list
    df_text['page'] = line_page_list
    df_text['line_number'] = lines
    df_text['ascii_ord'] = ascii_ord
    df_text['hash'] = df_text['sentences'].apply(hashify)
    df_text['left_x'] = bottom_left_x
    df_text['left_y'] = bottom_left_y
    ## Identify  Section Type for each page
    # sec_name_list = get_page_sections_from_list(line_sentence_list)
    # print("sec_name_list",len(sec_name_list),len(bottom_left_y),len(line_sentence_list))
    # df_text['SecName']= sec_name_list 
    df['space'] = space_list
    df['distance'] = distance_list
    # df = df.sort_values(by=['line_number', 'Top_left_x', 'Bottom_right_y'], ascending = [True, True, True]).reset_index(drop = True)    
    # print(df_text.columns.tolist())
    ## Identify  Section Type for each page
    df_text = get_page_sections(df_text)

    df, df_text = adding_vertical_bottom_space(df, df_text)

    # Get dictionary of header and footers with key as page number and values are header and footer line number
    header_dict, footer_dict = header_footer_dictionary(df_text)

    my_footer_list = []
    for page, footer_list in footer_dict.items():
        if footer_list and df_text[df_text['page']==page]['sentences'].tolist() :
            footer = df_text[df_text['page']==page]['sentences'].tolist()[-1]
            my_footer_list.append(footer)

    is_hierarchy = True
    my_keyword_form = ['DD Form']
    for keyword in my_keyword_form:    
        my_sum = sum([keyword  in my_f for my_f in my_footer_list])
        if my_sum > len(my_footer_list)/2:
            is_hierarchy = False

    df_text_header_footer_removal = remove_header_footer(df_text, header_dict, footer_dict)
    toc = time.time()
    header_footer_removing_time = toc-tic

    tic = time.time()
    random_n_pdfs = randomize(l=df['page'].unique().tolist(), top_n=CHECK_N_PAGES)

    count_list = []
    for df_page in [df[df['page'] == i] for i in random_n_pdfs]:            
        is_split = determine_split(df_page, is_scanned = is_scanned)
        count_list.append(is_split)

    count = sum(count_list)

    total_pages = len(df['page'].unique())

    if total_pages > CHECK_N_PAGES and count and count in range(CHECK_N_PAGES-2, CHECK_N_PAGES+1):
        PAGE_SPLIT = True
        print("{} out of {} pages have page split".format(count, CHECK_N_PAGES))
    elif CHECK_N_PAGES > total_pages and count and count in range(total_pages-2, total_pages+1):
        PAGE_SPLIT = True
        print("{} out of {} pages have page split".format(count, total_pages))
    else:
        PAGE_SPLIT = None

    # PAGE_SPLIT = None
    if PAGE_SPLIT:
        df_text_two_column_support = split(df=df, df_text_input=df_text_header_footer_removal, is_scanned=is_scanned)
    else:            
        df_text_two_column_support = df_text
        
    # Get dictionary of header and footers with key as page number and values are header and footer line number
    header_dict, footer_dict = header_footer_dictionary(df_text_two_column_support)
    df_text_two_column_support_header_footer_removal = remove_header_footer(df_text_two_column_support, header_dict, footer_dict)
    toc = time.time()
    determing_and_splitting_two_column_document_time = toc - tic
    df_line = calculate_paragraph_and_column(df_line)
    azureOCR = OCR( file_name = file_path, 
                    ocr_response = ocr_response,
                    df = df,
                    df_text = df_text,
                    df_text_header_footer_removal = df_text_header_footer_removal,
                    is_scanned = is_scanned,
                    df_text_two_column_support = df_text_two_column_support,
                    df_text_two_column_support_header_footer_removal = df_text_two_column_support_header_footer_removal,
                    df_line = df_line,
                    header_dict = header_dict,
                    footer_dict = footer_dict,
                    PAGE_SPLIT=PAGE_SPLIT,
                    is_hierarchy = is_hierarchy,
                    total_time_of_post_request = total_time_of_post_request,
                    total_time_of_get_request = total_time_of_get_request,
                    ocr_parse_time = ocr_parse_time,
                    header_footer_removing_time = header_footer_removing_time,
                    determing_and_splitting_two_column_document_time=determing_and_splitting_two_column_document_time
                    )
    
    print("FINISHED GENERATING AZURE READ RESULTS")
    return azureOCR


def split(df, df_text_input, is_scanned):
    print("azure_ocr.is_scanned", is_scanned)
    if is_scanned:
        space_threshold = 140
        split_threshold = 70
    else:
        space_threshold = 0.07
        split_addon = 0.4
        
    df_text = pd.DataFrame()
    split_threshold = max(df['Bottom_right_x'])/2
    d = df[df['space'] > space_threshold]
    if not d.empty:
        skip_line_numbers = min(d['line_number'])
        d = d[d['Bottom_right_x'] > split_threshold]
        if not d.empty and skip_line_numbers <= 10:
            df_split_threshold = min(d['Bottom_right_x']) + split_addon 
            df['right_side'] = df['Bottom_right_x'] > df_split_threshold 
            my_list = df['right_side'].tolist()
            
            for line_number in range(skip_line_numbers):
                temp_df = df[df["line_number"]==line_number]
                df.loc[temp_df.index, 'right_side'] = None
            else:
                left_side_index = df[df['right_side']==0].index
                right_side_index = df[df['right_side']==1].index
                left_dataframe = df.loc[left_side_index, :]
                right_dataframe = df.loc[right_side_index, :]

            sentence_list = []
            line_page_list = []
            line_number_list = []
            bottom_left_x = []
            bottom_left_y = []
            for page in sorted(set(df.page)):
                left_page_dataframe = left_dataframe[left_dataframe['page']==page]
                right_page_dataframe = right_dataframe[right_dataframe['page']==page]
                
                left_sentence_list = []
                left_line_page_list = []
                left_line_number_list = []
                
                right_sentence_list = []
                right_line_page_list = []
                right_line_number_list = []
                left_x = []
                left_y = []
                for line_number in sorted(set(right_page_dataframe['line_number']).union(set(left_page_dataframe['line_number']))):
                    if line_number in set(df_text_input['line_number']):
                        left_df = left_page_dataframe[left_page_dataframe['line_number'] == line_number]
                        if not left_df.empty:
                            left_line = ' '.join(left_df['Description'].tolist())
                            left_sentence_list.append(left_line)
                            left_line_page_list.append(page)
                            left_line_number_list.append(line_number)
                            left_x.append(sorted(left_df['Bottom_left_x'].unique())[0])
                            left_y.append(sorted(left_df['Bottom_left_y'].unique())[0])
                        right_df = right_page_dataframe[right_page_dataframe['line_number'] == line_number]
                        if not right_df.empty:
                            right_line = ' '.join(right_df['Description'].tolist())
                            right_sentence_list.append(right_line)
                            right_line_page_list.append(page)
                            right_line_number_list.append(line_number)
                            left_x.append(sorted(right_df['Bottom_left_x'].unique())[0])
                            left_y.append(sorted(right_df['Bottom_left_y'].unique())[0])
                else:
                    sentence_list.extend(left_sentence_list)
                    sentence_list.extend(right_sentence_list)

                    line_page_list.extend(left_line_page_list)
                    line_page_list.extend(right_line_page_list)
                                
                    line_number_list.extend(left_line_number_list)
                    line_number_list.extend(right_line_number_list)
                    bottom_left_x.extend(left_x)
                    bottom_left_y.extend(left_y)
            
            ascii_ord = [[ord(letter) for letter in line_sentence] for line_sentence in sentence_list]

            df_text = pd.DataFrame()
            df_text['sentences'] = sentence_list
            df_text['page'] = line_page_list
            df_text['line_number'] = line_number_list #range(1, len(line_number_list)+1)
            df_text['ascii_ord'] = ascii_ord
            df_text['hash'] = df_text['sentences'].apply(hashify)
            df_text['left_x'] = bottom_left_x
            df_text['left_y'] = bottom_left_y
        
            if df_text.empty:
                df_text = df_text_input
            if len(df_text.loc[df_text.index[-1],:].tolist().pop(0)) == 1:
                df_text = df_text.drop(df_text.index[-1])
        else:
            df_text = df_text_input
    else:
        df_text = df_text_input

    df, df_text = adding_vertical_bottom_space(df, df_text)
    return df_text


def remove_header_footer(df_text, header_dict, footer_dict):
    df_text_header_footer_removal = pd.DataFrame()
    for page_num in sorted(set(df_text['page'])):
        page_df_text = df_text[df_text['page'] == page_num]
        
        if page_num in header_dict.keys() and header_dict[page_num]:
            start_index = max(header_dict[page_num])+1
        else:
            start_index = 0
        
        if page_num in footer_dict.keys() and footer_dict[page_num]:
            end_index = NUM_LINE_FOOTER - min(footer_dict[page_num])
        else:
            end_index = None
            
        if end_index is None:
            final_df_index = page_df_text.index[start_index:]
        else:
            final_df_index = page_df_text.index[start_index:-end_index]
        final_df = page_df_text.loc[final_df_index]
        # df_text_header_footer_removal = df_text_header_footer_removal.append(final_df)
        df_text_header_footer_removal =pd.concat([df_text_header_footer_removal, final_df],axis=0,ignore_index=True)


    return df_text_header_footer_removal


def header_footer_dictionary(df_text):
    df_text['ascii_sum'] = df_text['ascii_ord'].apply(lambda x: sum(x))
    grouped = df_text.groupby('page')
    header_list = []
    footer_list = []
    for page, group in grouped:  
        h_index = group.iloc[:NUM_LINE_HEADER].index
        f_index = group.iloc[-NUM_LINE_FOOTER:].index
        h_df = df_text.loc[h_index, 'hash']
        f_df = df_text.loc[f_index, 'hash']
        header_list.append(h_df.tolist()+[page])
        footer_list.append(f_df.tolist()+[page])
    else:
        # HEADER
        h_df = pd.DataFrame(header_list)
        h_diff_df=h_df.diff()
        # h_diff_df = h_diff_df.iloc[1:,:].append(h_diff_df.iloc[-1, :])
        h_diff_df = pd.concat([h_diff_df.iloc[1:,:],pd.DataFrame(h_diff_df.iloc[-1, :]).T])
        h_diff_df = h_diff_df.reset_index(drop=True)
        # assign line number in index 5
        h_diff_df[NUM_LINE_HEADER] = h_df[NUM_LINE_HEADER]
        h_diff_df = h_diff_df[~((0 <= h_diff_df) & (h_diff_df < HEADER_FOOTER_THRESHOLD))]
        h_diff_df.index = h_df[NUM_LINE_HEADER].tolist()
        h_diff_df = h_diff_df.iloc[1:,:-1]    
        header_dict = {}
        for page, lines in zip(h_diff_df.index, h_diff_df.values):
            h_list = []
            for line_num, line in enumerate(lines):
                if pd.isna(line):
                    if not h_list:
                        h_list.append(line_num)
                    elif h_list[-1]+1 == line_num:
                        h_list.append(line_num)
                    else:
                        break
            header_dict[page] = h_list
        else:
            reg_value = None
            for page_key in sorted(header_dict.keys()):
                header_dict[page_key] = header_dict[page_key] and list(range(max(header_dict[page_key])+1))
                if not header_dict[page_key] and not reg_value:
                    continue
                elif header_dict[page_key] and not reg_value:
                    reg_value = header_dict[page_key] 
                    continue
                else:
                    # for reg_val is not None
                    if len(header_dict[page_key]) == len(reg_value):
                        reg_value = header_dict[page_key]
                    elif len(header_dict[page_key]) < len(reg_value):
                        temp = header_dict[page_key]
                        header_dict[page_key] = reg_value
                        reg_value = temp
                    else:
                        reg_value = header_dict[page_key] 
        
                
        # FOOTER
        f_df = pd.DataFrame(footer_list)
        f_diff_df=f_df.diff()
        f_diff_df[NUM_LINE_FOOTER] = f_df[NUM_LINE_FOOTER]
        f_diff_df = f_diff_df[~((0 <= f_diff_df) & (f_diff_df < HEADER_FOOTER_THRESHOLD))]
        f_diff_df.index = f_df[NUM_LINE_FOOTER]
        f_diff_df = f_diff_df.iloc[1:,:-1]    
        footer_dict = {}
        for page, lines in zip(f_diff_df.index, f_diff_df.values):
            f_list = []
            for line_num, line in enumerate(lines):
                if pd.isna(line):
                    if not f_list:
                        f_list.append(line_num)
                    elif f_list[-1] == line_num - 1:
                        f_list.append(line_num)
                    else:
                        break                    

            footer_dict[page] = f_list
        else:
            for page_key in sorted(footer_dict.keys()):
                footer_dict[page_key] = footer_dict[page_key] and list(range(min(footer_dict[page_key]), NUM_LINE_FOOTER))
  
            
            
    return header_dict, footer_dict


def determine_split2(df):
    data = pd.DataFrame()
    data['Description'] = df['Description']
    data['old_Bottom_left_x'] = df['Bottom_left_x']
    data['Bottom_left_x'] = df['Bottom_left_x'].apply(lambda x: round(x,2))
    data['old_Bottom_right_x'] = df['Bottom_right_x']
    data['Bottom_right_x'] = df['Bottom_right_x'].apply(lambda x: round(x,2))
    data['page'] = df['page']
    data['line_number'] = df['line_number']


def determine_split(df, is_scanned):    
    df = df.copy()
    if is_scanned:
        d = df[df['space'] > 140]
        split_threshold = (max(df['Bottom_right_x']) - min(df['Bottom_left_x']))/2
    else:
        split_threshold = max(df['Bottom_right_x'])/2
        d = df[(5 > df['space']) & (df['space'] > 0.51)]
    df['middle_x'] = min(df['Bottom_left_x']) + split_threshold        
    count = 0
    for i in d.index:
        if i != df.index[-1] and df.loc[i, 'Bottom_right_x'] < df.loc[i, 'middle_x']  < df.loc[i+1, 'Bottom_right_x']:
            count+=1
    return count > len(d)/2


def randomize(l, top_n =7):
    if top_n > len(l):
        top_n = len(l)
    
    random_list = []
    length = len(l)
    for i in range(top_n):
        i = (i*-1)
        if abs(i%2) == 1:
            length = length//2
            random_list.append(l[int(length * i/-i)])
        else:
            if i:
                random_list.append(l[int(length * i/i)])
            else:
                random_list.append(l[int(length * i)])
    return random_list


def adding_vertical_bottom_space(df, df_text):
    Bottom_right_y_list = []
    Top_left_y_list = []
    all_lines = sorted(set(df['line_number']))
    for line_number, next_line_number in zip(all_lines[:-1], all_lines[1:]):
        line_df = df[df['line_number'] == line_number]
        Bottom_right_y = max(line_df['Bottom_right_y'].mode())
        Bottom_right_y = Bottom_right_y and [Bottom_right_y] * len(line_df)
        Bottom_right_y_list.extend(Bottom_right_y)
 
        next_line_df = df[df['line_number'] == next_line_number]
        Top_left_y = min(next_line_df['Top_left_y'].mode())
        Top_left_y = Top_left_y and [Top_left_y] * len(line_df)
        # print("len_line df",len(line_df),Top_left_y)
        if Top_left_y:
            Top_left_y_list.extend(Top_left_y)

        else:
            print("In else")
            # print(len(Bottom_right_y),"df_length",len(line_df))
            Top_left_y = [None]*len(line_df)
            Top_left_y_list.extend(Top_left_y)
        # print(Top_left_y)
        # Top_left_y_list.extend(Top_left_y)
        
        assert len(line_df) == len(Top_left_y) == len(Bottom_right_y)
    else:
        padding = len(df) - len(Top_left_y_list)
        Bottom_right_y_list.extend([None]*padding)
        Top_left_y_list.extend([None]*padding)
        
        
    df['Bottom_right_y_list'] = Bottom_right_y_list
    df['Top_left_y_list'] =  Top_left_y_list
    df['vertical_bottom_space'] = df['Top_left_y_list']  - df['Bottom_right_y_list'] 
    
    vertical_bottom_space_df_text_list = []
    for line_number in sorted(df_text['line_number']):
        my_df = df[df['line_number'] == line_number]
        vertical_bottom_space = my_df['vertical_bottom_space'].unique().tolist()
        vertical_bottom_space = vertical_bottom_space and vertical_bottom_space.pop(0)
        vertical_bottom_space_df_text_list.append(vertical_bottom_space)
    else:
        df_text['vertical_bottom_space'] = vertical_bottom_space_df_text_list 
    
    return df, df_text

def get_page_sections_from_list(list_sentences):
    sec_name_list = []
    sec_name=''
    for index,each_line in enumerate(list_sentences):
        # print("each_line get_page_sections_from_list",each_line)
        section_value = get_line_section(each_line)
        if section_value:
            sec_name_list.append(section_value)
            sec_name = section_value
        else:
            sec_name_list.append(sec_name)
    return sec_name_list

def get_page_sections(df_text_data):
    sec_name_df = pd.Series()
    df_text_data['sec_name'] = ''
    sec_name = ''
    # for i in range(len(df_text)):   
    for index,row in df_text_data.iterrows():
        section_value = get_line_section(row['sentences'])
        if section_value:
            sec_name_df.loc[index] = section_value
            sec_name = section_value
        else:
            sec_name_df.loc[index]=sec_name
    df_text_data['sec_name'] = sec_name_df
    return df_text_data

def get_line_section(line_dat):
    match_value=None
    if isinstance(line_dat,str):
        # print(line_dat)
        # line_dat = line_dat.decode('utf-8')
        for each_form_type in  section_pattern_dic:
            form_reg = section_pattern_dic[each_form_type]
            matcs = re.search(form_reg,line_dat) 
            # matcs =form_reg.findall(page_text_data_up)
            if matcs:
                p_match = matcs.string[matcs.span()[0]: matcs.span()[1]]
                # print(p_match,each_form_type)
                match_value=each_form_type
    return match_value


# def get_splitpdfs_smartresponse(file_name_list,page_params=None):
    
#     total_time_of_get_request =0
#     ocr_parse_time = 0
#     details =[]
#     details_line_by_line =[]
#     last_page =0
#     total_time_of_get_request =0
#     total_time_of_post_request =0 
#     for ind, each_filename in enumerate(file_name_list):

#         fd = open(each_filename, "rb")
#         file_data = fd.read()
#         # Set Content-Type to octet-stream
#         headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
#         # put the byte array into your post request
#         tic = time.perf_counter()
#         #params = {'pages' : '1-2'}
#         print("page_params",page_params)
#         ocr_response = requests.post(text_recognition_url, headers=headers, params=page_params, data = file_data)
#         toc = time.perf_counter()
#         total_time_of_post_request_split =  toc-tic
#         print("{} seconds required for Azure READ API POST request\n".format(total_time_of_post_request_split))
#         total_time_of_post_request = total_time_of_post_request + total_time_of_post_request_split

#         while not ocr_response.ok:
#             ocr_response = requests.post(text_recognition_url, headers=headers, params=page_params, data = file_data)
#             ocr_response.raise_for_status()
#         # closing file
#         fd.close()

#         if ind==0:
#             details,details_line_by_line,current_page,total_time_of_get_request_new = smart_response_multiplesplit_pfds(ocr_response,headers,ind,details,details_line_by_line,ind)
#             last_page = current_page
#             total_time_of_get_request = total_time_of_get_request + total_time_of_get_request_new
#         else:
#             print("last_page",last_page)
#             details,details_line_by_line,current_page,total_time_of_get_request_new = smart_response_multiplesplit_pfds(ocr_response,headers,ind,details,details_line_by_line,last_page)
#             last_page = current_page 
#             total_time_of_get_request = total_time_of_get_request + total_time_of_get_request_new

#     return details,details_line_by_line,total_time_of_get_request,ocr_response,total_time_of_post_request

def smart_response_multiplesplit_pfds(response, headers,ind,details,details_line_by_line,last_pag):            
    # Extracting text requires two API calls: One call to submit the
    # image for processing, the other to retrieve the text found in the image.
    
    # Holds the URI used to retrieve the recognized text.
    operation_url = response.headers["Operation-Location"]
    
    # The recognized text isn't immediately available, so poll to wait for completion.
    analysis = {}
    poll = True
    tic = time.perf_counter()
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        
        # print(json.dumps(analysis, indent=4))
    
        time.sleep(1)
        if ("analyzeResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'failed'):
            poll = False
    else:
        toc = time.perf_counter()
        total_time_of_get_request = toc-tic
        print("{} seconds required for Azure READ API GET request\n".format(total_time_of_get_request))
    current_page = 0
    if ind==0:
        page_offset =0
    else:
        page_offset = last_pag
    tic = time.time()
    #details = []
    #details_line_by_line = []
    if ("analyzeResult" in analysis):
        # Extract the recognized text, with bounding boxes. 
        for read_result in analysis["analyzeResult"]["readResults"]:
            for key, value in read_result.items():
                if key == 'page':
                    page = value + page_offset
                    current_page = page
                if key == 'angle':
                    angle = value
                if key == 'width':
                    width = value
                if key == 'height':
                    height = value
                if key == 'unit':
                    unit = value
                if key == 'lines':
                    lines = value
                    
            for line in lines:
                Top_left_x, Top_left_y, Top_right_x, Top_right_y, Bottom_right_x, Bottom_right_y, Bottom_left_x, Bottom_left_y = line['boundingBox']
                Description = line['text']       
                details_line_by_line.append([Description, Top_left_x, Top_left_y, Top_right_x, Top_right_y, Bottom_right_x, Bottom_right_y, Bottom_left_x, Bottom_left_y ] + [page, angle, width, height, unit])
                inner_details = []
                for word in line['words']:
                    inner_details.append([word['text'], *word["boundingBox"], word['confidence']] + [page, angle, width, height, unit])
                else:
                    details.extend(inner_details)
    time.sleep(10)
    return details,details_line_by_line,current_page,total_time_of_get_request

def combine_smartresponse_to_df(details,details_line_by_line):
    tic = time.time()
    df = pd.DataFrame(details, columns = ['Description', 
                                'Top_left_x','Top_left_y' ,
                                'Top_right_x', 'Top_right_y' ,
                                'Bottom_right_x','Bottom_right_y',
                                'Bottom_left_x','Bottom_left_y', "Confidence",
                                "page", "angle", "width", "height", "unit"])
    df, is_scanned = line_numbers(df)

    df_line = pd.DataFrame(details_line_by_line, columns = ['Description', 
                            'Top_left_x','Top_left_y' ,
                            'Top_right_x', 'Top_right_y' ,
                            'Bottom_right_x','Bottom_right_y',
                            'Bottom_left_x','Bottom_left_y',
                            "page", "angle", "width", "height", "unit"])
    df_line, is_scanned = line_numbers(df_line)
    toc = time.time()
    ocr_parse_time = toc-tic

    return df, df_line, is_scanned, ocr_parse_time

    
def generate_dfline(details_list):
    df_line = pd.DataFrame(details_list, columns = ['Description', 
    'Top_left_x','Top_left_y' ,'Top_right_x', 'Top_right_y' ,'Bottom_right_x','Bottom_right_y',
    'Bottom_left_x','Bottom_left_y',"page", "angle", "width", "height", "unit"])

    df_line, is_scanned = line_numbers(df_line)

    df_line = calculate_paragraph_and_column(df_line)

    return df_line


# def get_line_section(line_dat):
#     print(line_dat)
#     line_dat = line_dat.decode('utf-8')
#     match_value=''
#     for each_form_type in  section_pattern_dic:
#         form_reg = section_pattern_dic[each_form_type]
#         matcs = re.search(form_reg,line_dat) 
#         # matcs =form_reg.findall(page_text_data_up)
#         if matcs:
#             p_match = matcs.string[matcs.span()[0]: matcs.span()[1]]
#             print(p_match,each_form_type)
#             match_value=each_form_type
#     return match_value


class AzureOcr:
    subscription_key = BaseConfig.READ_API_SUBSCRIPTION_KEY
    endpoint = BaseConfig.READ_API_ENDPOINT
    text_recognition_url = endpoint + "/vision/v3.2/read/analyze"

    def __init__(self,file_path,page_params=None,timeout=None):
        self.file_path = file_path
        self.page_params=page_params
        self.timeout = timeout


    def __gettextfromazure(self):  
        print()
        status_dict = {"status": False}
        print("Hitting Azur READ API OCR")
        
        print("READ_API_ENDPOINT:", self.endpoint)
        print()
        print("READ_API_SUBSCRIPTION_KEY:", self.subscription_key)
        pdf_reader = PdfFileReader(open(self.file_path, "rb"),strict=False)
        total_pages = pdf_reader.numPages
        print("Total Pages for READ API is {}".format(total_pages))
        fd = open(self.file_path, "rb")
        file_data = fd.read()
        # Set Content-Type to octet-stream
        key_args_dict ={'timeout':self.timeout}
        print("key_args_dict",key_args_dict)
        headers = {'Ocp-Apim-Subscription-Key': self.subscription_key, 'Content-Type': 'application/octet-stream'}
        # put the byte array into your post request
        tic = time.perf_counter()
        #params = {'pages' : '1-2'}
        print("page_params",self.page_params,self.text_recognition_url)
        try:

            ocr_response = requests.post(self.text_recognition_url, headers=headers, params=self.page_params, data = file_data,**key_args_dict)
            toc = time.perf_counter()
            total_time_of_post_request = toc-tic
            print("{} seconds required for Azure READ API POST request\n".format(total_time_of_post_request))
            print(ocr_response.raise_for_status())
            print("STATUS CODE",ocr_response.status_code)

            # while not ocr_response.ok:
            #     ocr_response = requests.post(self.text_recognition_url, headers=headers, params=self.page_params, data = file_data)
            #     ocr_response.raise_for_status()
            # closing file
            fd.close()
        except Exception as e:
            status_dict = {"status": False}
            status_dict['error']=str(e.__class__.__name__)
            status_dict['message']=str(e)
            print("Error Name: ", e.__class__.__name__)
            print("Error Message: ", str(e))
            is_scanned=False
            dataframes_tuple={}
            return is_scanned,dataframes_tuple,status_dict
        
        df_azure, df_line_azure, is_scanned,total_time_of_get_request,ocr_parse_time = self.__smart_response(ocr_response, headers)
        dataframes_tuple = generate_dataframes(df_azure,df_line_azure,is_scanned,self.file_path,ocr_response,total_time_of_post_request,total_time_of_get_request,ocr_parse_time)
        status_dict = {"status": True}
        status_dict['error']=''
        status_dict['message']=''
       
        return is_scanned,dataframes_tuple,status_dict

        
    def __smart_response(self,response, headers):            
        # Extracting text requires two API calls: One call to submit the
        # image for processing, the other to retrieve the text found in the image.
        
        # Holds the URI used to retrieve the recognized text.
        operation_url = response.headers["Operation-Location"]
        
        # The recognized text isn't immediately available, so poll to wait for completion.
        analysis = {}
        poll = True
        tic = time.perf_counter()
        while (poll):
            response_final = requests.get(
                response.headers["Operation-Location"], headers=headers)
            # print("response_final",response_final)
            analysis = response_final.json()
            
            # print(json.dumps(analysis, indent=4))
        
            time.sleep(1)
            if ("analyzeResult" in analysis):
                poll = False
            if ("status" in analysis and analysis['status'] == 'failed'):
                poll = False
        else:
            toc = time.perf_counter()
            total_time_of_get_request = toc-tic
            print("{} seconds required for Azure READ API GET request\n".format(total_time_of_get_request))
        
        tic = time.time()
        details = []
        details_line_by_line = []
        if ("analyzeResult" in analysis):
            # Extract the recognized text, with bounding boxes. 
            for read_result in analysis["analyzeResult"]["readResults"]:
                for key, value in read_result.items():
                    if key == 'page':
                        page = value
                    if key == 'angle':
                        angle = value
                    if key == 'width':
                        width = value
                    if key == 'height':
                        height = value
                    if key == 'unit':
                        unit = value
                    if key == 'lines':
                        lines = value
                        
                for line in lines:
                    Top_left_x, Top_left_y, Top_right_x, Top_right_y, Bottom_right_x, Bottom_right_y, Bottom_left_x, Bottom_left_y = line['boundingBox']
                    Description = line['text']       
                    details_line_by_line.append([Description, Top_left_x, Top_left_y, Top_right_x, Top_right_y, Bottom_right_x, Bottom_right_y, Bottom_left_x, Bottom_left_y ] + [page, angle, width, height, unit])
                    inner_details = []
                    for word in line['words']:
                        inner_details.append([word['text'], *word["boundingBox"], word['confidence']] + [page, angle, width, height, unit])
                    else:
                        details.extend(inner_details)
                
        
        df = pd.DataFrame(details, columns = ['Description', 
                                'Top_left_x','Top_left_y' ,
                                'Top_right_x', 'Top_right_y' ,
                                'Bottom_right_x','Bottom_right_y',
                                'Bottom_left_x','Bottom_left_y', "Confidence",
                                "page", "angle", "width", "height", "unit"])
        df, is_scanned = line_numbers(df)
        
        df_line = pd.DataFrame(details_line_by_line, columns = ['Description', 
                                'Top_left_x','Top_left_y' ,
                                'Top_right_x', 'Top_right_y' ,
                                'Bottom_right_x','Bottom_right_y',
                                'Bottom_left_x','Bottom_left_y',
                                "page", "angle", "width", "height", "unit"])
        df_line, is_scanned = line_numbers(df_line)
        toc = time.time()
        ocr_parse_time = toc-tic

        return df, df_line, is_scanned,total_time_of_get_request,ocr_parse_time

    def main(self):
        isREADAPIfailed=False
        try:
            is_scanneds,dataframes_tuplse,status_dict = AzureOcr.__gettextfromazure(self)
            if status_dict.get('status'):
                print("MAIN DONE",status_dict)
            else:
                print("READ API FAILED WITH STATUS")
                print(status_dict)
                isREADAPIfailed=True
                is_scanneds,dataframes_tuplse = False, {}

        except Exception as e:
            isREADAPIfailed=True
            print("Error in Azure READ API", str(e))
            is_scanneds,dataframes_tuplse = False, {}
        return is_scanneds,dataframes_tuplse,isREADAPIfailed

if __name__ == "__main__":
    azure_ocr_obj = AzureOcr('/Users/shyamz/TM/revamp/tm-ici-app/temp/am624/splitpages_first3.pdf')
    is_scanneds,dataframes_tuplse,isREADAPIfailed= azure_ocr_obj.main()
    print(is_scanneds,dataframes_tuplse)