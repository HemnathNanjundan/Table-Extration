from shapely.geometry import Polygon
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import os
def gen_polygon(cords):
    x,y,w,h = cords
    # top_left = (x, y) top_right = (x+w, y) bottom_right = (x+w, y+h) bottom_left = (x, y+h)
    bbox_vertices =[(x, y), (x+w, y),(x+w, y+h),(x+w, y+h),(x, y+h)]
    # bbox_vertices = [(int(row['Top_left_x']), int(row['Top_left_y'])), (int(row['Bottom_right_x']), int(row['Top_left_y'])),
    #                     (int(row['Bottom_right_x']), int(row['Bottom_right_y'])), (int(row['Top_left_x']), int(row['Bottom_right_y']))]
    # Create a Shapely Polygon object
    box_poly = Polygon(bbox_vertices)
    return box_poly
def convert_poly_tocords(ply):
    X, Y = ply.exterior.xy
    x = int(X[0])
    y = int(Y[0])
    x1 = int(X[2])
    y1 = int(Y[2])
    w = y1-y
    h=x1-x
    return (x, y, w, h)

def get_boxes_from_header_csv(bbox_data):
    # bbox_data = pd.read_csv('/Users/shyamz/Downloads/bugs_data/40860/results_working/3files/of-347-15-pages/outputs/header_77.csv')
    # new_data1 = bbox_data[['Top_left_x', 'Top_left_y', 'Bottom_right_x', 'Bottom_right_y']].mul(300)

    bboxes_list = []
    # Iterate over the dataframe rows
    for index, row in bbox_data.iterrows():
        # Extract bounding box vertices
        # bbox_vertices = [(int(row['Top_left_x']*300), int(row['Top_left_y']*300)),
        #                 (int(row['Bottom_right_x']*300), int(row['Top_left_y']*300)),
        #                 (int(row['Bottom_right_x']*300), int(row['Bottom_right_y']*300)), 
        #                 (int(row['Top_left_x']*300), int(row['Bottom_right_y']*300))]
        
        box_vertices = (int(row['Top_left_x']*300), int(row['Top_left_y']*300),
                        int(row['Bottom_right_x']*300) - int(row['Top_left_x']*300),
                        int(row['Bottom_right_y']*300)- int(row['Top_left_y']*300))

        text = row['Description']
        # Create a Shapely Polygon object
        # poly = Polygon(bbox_vertices)
        bboxes_list.append([text,box_vertices])
    # line_number =77
    return bboxes_list


class Table_Drawer:
  def __init__(self,pdf,list_num,temp_dir,dpi=300):
    self.pdf=pdf
    self.image=None
    self.page_num=list_num
    self.line_cords=[]
    self.table_contours=None
    self.upper=None
    self.lower=None
    self.mimg=None
    self.leftmost_vertical_line=None
    self.rightmost_vertical_line=None
    self.map={}
    self.save_path=temp_dir
    self.dpi=dpi
    self.results = {}

  def Extract(self):
    # images = convert_from_path(self.pdf,dpi=self.dpi)
    # for i in range(0,len(images)):
    for i in self.page_num:
        try:
            
          images_from_path = convert_from_path(self.pdf,dpi=self.dpi, output_folder=self.save_path,fmt="jpeg",output_file='page'+ str(i) +'.jpg',paths_only=True, first_page = i, last_page = i)
          # if i+1 in self.page_num:
            # with open(self.pdf[:-4]) as f:
            #   f.write(images[i].save(self.pdf[:-4]+'_'+ 'page'+ str(i) +'.jpg', 'JPEG'))
            # file_path_name = os.path.join(self.save_path,'_'+ 'page'+ str(i) +'.jpg')
            # images_from_path[i].save(file_path_name, 'JPEG')
          self.image = images_from_path[0]
          self.results[str(i)]={}
          self.plot(i)
          self.Hplot(i)
          self.masking(i)
          self.boundary(self.b_x0,self.b_y0,self.b_x1,self.b_y1,i)
          # self.Tstructure()
          self.Tcontuors(i)
          self.center(i)
          self.grouping_rows(i)
          self.grouping_cols(i)
          self.map={}
          self.line_cords=[]
          self.table_contours=None
          print("Page {} is completed".format(i))
        except:
          print("Page {} is Failed".format(i))
    return self.results
  def plot(self,page_num):
    image = cv2.imread(self.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Find contours and iterate through them
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_cords=[]
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the start and end coordinates as circles
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(image, (x + w, y + h), 5, (0, 0, 255), -1)
        # Display the coordinates as text
        cv2.putText(image, f"({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(image, f"({x + w}, {y + h})", (x + w, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Get the bounding rectangle of the contour
        # Append the coordinates to the list
        v_cords.append(((x, y), (x + w, y + h)))
    # filename=os.path.join(self.save_path,'_page_{0}_VLines.jpg'.format(page_num))
    # cv2.imwrite(filename, image)
    maxline=self.longest_line(v_cords)
    # print(maxline,"\n\n\n")
    # for i, coordinates in enumerate(v_cords):
    #   start = coordinates[0]
    #   end = coordinates[1]
    #   self.line_cords.append([start,end])
    #   # print(f"{i + 1} - {start},{end}")
    # print(v_cords,"Before \n\n\n\n\n")

    v_cords = self.filterV(v_cords,maxline,page_num)
    for i, coordinates in enumerate(v_cords):
          start = coordinates[0]
          end = coordinates[1]
          self.line_cords.append([start,end])
    sorted_lines = sorted(v_cords, key=lambda line: line[0][0])

    self.leftmost_vertical_line = sorted_lines[0]
    self.rightmost_vertical_line = sorted_lines[-1]
    # print(v_cords,"\n\n\n\n\n")
    # filename=os.path.join(self.save_path,'_page_{0}_vertical_lines.jpg'.format(page_num))
    # cv2.imwrite(filename, image)

  def Tcontuors(self,page_num):
      image_with_boxes = self.mimg
      edges = cv2.Canny(self.mimg, 100, 200)  # Adjust the threshold values as needed
      self.table_contours = []
      min_area = 150  # Minimum contour area to consider as a table
      min_height = 10  # Minimum height of the table to avoid small artifacts
      max_area = 2000*2000
      contours, _ = cv2.findContours(image_with_boxes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      for contour in contours:
          # print(len(contour))
          x, y, w, h = cv2.boundingRect(contour)
          if max_area > cv2.contourArea(contour) > min_area  and h > min_height and (x,y,w,h) not in self.table_contours:
              self.table_contours.append((x, y, w, h))

      # print(len(self.table_contours),"\n\n\n\n")
      colors = np.random.randint(0,255, size=(len(self.table_contours), 4), dtype=np.uint8)
      coordinates_dicts = {}

      for i, (x, y, w, h) in enumerate(self.table_contours):
          color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
          cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), color, 3)
          top_left = (x, y)
          top_right = (x+w, y)
          bottom_right = (x+w, y+h)
          bottom_left = (x, y+h)
          # coordinates_dicts[(x, y, w, h)]={"TL" : top_left,
          # "TR" : top_right,
          # "BR" : bottom_right,
          # "BL" : bottom_left,
          # "width" : w,
          # "height": h,
          # "isheader": False,
          # "Row":'', "column" :'',
          # "index":i}

          # Display the coordinates
          cv2.putText(image_with_boxes, f"TL: {top_left}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
          # cv2.putText(image_with_boxes, f"TR: {top_right}", (x+w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          # cv2.putText(image_with_boxes, f"BR: {bottom_right}", (x+w, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
          # cv2.putText(image_with_boxes, f"BL: {bottom_left}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          # print(f"Table {i+1} - Edge Coordinates:")
          # print("Top Left:", top_left)
          # print("Top Right:", top_right)
          # print("Bottom Right:", bottom_right)
          # print("Bottom Left:", bottom_left)
          # print()
      # print("coordinates_dicts",coordinates_dicts)
      # filename=os.path.join(self.save_path,'_page_{0}_Tcontours.jpg'.format(page_num))
      # cv2.imwrite(filename, image_with_boxes)
      # cv2_imshow(image_with_boxes)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # self.mimg=image_with_boxes
  def masking(self,page_num):
    image = cv2.imread(self.image)
    mask=np.zeros_like(image[:, :, 0])
    line_color = (255, 255, 255)  # White color for the lines
    line_thickness = 2

    # Draw multiple lines on the image
    for coords in self.line_cords:
        point1, point2 = coords
        cv2.line(mask, point1, point2, line_color, line_thickness)
    # filename=os.path.join(self.save_path,'_page_{0}_allLines.jpg'.format(page_num))
    # cv2.imwrite(filename, mask)
    self.mimg=mask

  def Hplot(self,page_num):
    image = cv2.imread(self.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    l=[]
    # Find contours and iterate through them for horizontal lines
    contours, _ = cv2.findContours(horizontal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_line_coordinates = []
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Append the coordinates to the list
        horizontal_line_coordinates.append(((x, y),(x + w, y + h)))
    sort=sorted(horizontal_line_coordinates, key=lambda coord: coord[0][1])
    self.upper=sort[0]
    self.lower=sort[-1]
    # print(self.Hupper,self.Hlower,"\n\n\n\n")
    print(sort)
    # Display the coordinates as separate outputs
    # for i, coordinates in enumerate(horizontal_line_coordinates):
    #     start = coordinates[0]
    #     end = coordinates[1]
    #     self.line_cords.append([start,end])
    #     # print(f"{i + 1} - {start},{end}")
    # print(self.line_cords)
# Draw circles and display the image with coordinates
    for coordinates in horizontal_line_coordinates:
        start = coordinates[0]
        end = coordinates[1]
        cv2.circle(image, start,5,(255, 0, 0), -1)
        cv2.circle(image, end,5,(0, 255, 255), -1)
        cv2.putText(image, f" {start}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 1)
        cv2.putText(image, f" {end}", (end[0] - 70, end[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)

    # filename=os.path.join(self.save_path,'_page_{0}_HLines.jpg'.format(page_num))
    # cv2.imwrite(filename, image)

    maxline=self.longest_line(horizontal_line_coordinates)
    # print(maxline,'\n\n\n')
    horizontal_line_coordinates = self.filterH(horizontal_line_coordinates,maxline,page_num)

    for i, coordinates in enumerate(horizontal_line_coordinates):
        start = coordinates[0]
        end = coordinates[1]
        self.line_cords.append([start,end])
        # print(f"{i + 1} - {start},{end}")
    # print(self.line_cords)
    # filename=os.path.join(self.save_path,'_page_{0}_vertical_lines.jpg'.format(page_num))
    # cv2.imwrite(filename, image)

  def center(self,page_num):
    self.mid_points=[]
    image=cv2.imread(self.image)

    def find_midpoint(rectangle):
        x, y, w, h = rectangle
        midpoint_x = x + (w // 2)
        midpoint_y = y + (h // 2)
        return (midpoint_x, midpoint_y)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to convert the image to binary
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    colors = np.random.randint(0,255, size=(len(self.table_contours), 4), dtype=np.uint8)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(self.mimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in self.table_contours:

      x, y, w, h = contour

      # Find the midpoint of the bounding box
      midpoint = find_midpoint((x, y, w, h))
      self.map[contour]=midpoint
      # self.mid_points.append(midpoint)
      # Draw a circle at the midpoint on the image
      cv2.rectangle(image, (x, y), (x+w, y+h),(0,255,0), 3)
      cv2.circle(image, midpoint, 5, (0, 0, 255), -1)

    # print(self.map,"\n\n\n\n\n")
    # Display the image with contours
    # filename=os.path.join(self.save_path,'_page_{0}_table_centres.jpg'.format(page_num))
    # cv2.imwrite(filename, image)

  def grouping_rows(self,page_num):
    dict_rows={}
    target_y=[]
    for contour in self.map:
      if self.map[contour][1] not in target_y:
        target_y.append(self.map[contour][1])
      if self.map[contour][1] in dict_rows:
        dict_rows[self.map[contour][1]].append(contour)
      else:
        dict_rows[self.map[contour][1]]=[contour]
    sorted_numbers = sorted(target_y)
    result=[]
    current_group = [sorted_numbers[0]]

    for i in range(1, len(sorted_numbers)):
        if abs(sorted_numbers[i] - sorted_numbers[i-1]) <= 5:
            current_group.append(sorted_numbers[i])
        else:
            result.append(current_group)
            current_group = [sorted_numbers[i]]
    result.append(current_group)
    f_contours=[]
    # print("dict_rows",dict_rows)
    for k in result:
      l=[]
      for h in k:
        for g in range(len(dict_rows[h])):
          l.append(dict_rows[h][g])
      f_contours.append(l)

    colors = np.random.randint(0,255, size=(len(self.table_contours), 4), dtype=np.uint8)
    image=cv2.imread(self.image)
    row_contours_list = []
    for d,f in enumerate(f_contours):

      color = (int(colors[d][0]), int(colors[d][1]), int(colors[d][2]))
      # print(d,"Rows", f)
      # d is number of row and f is list of bounding boxes
      row_list = []

      for val in f:
        x,y,w,h=val
        row_list.append(val)
        # row_list.append([d,val])

        cv2.rectangle(image, (x, y), (x+w, y+h),color, 3)
      row_list = sorted(row_list , key=lambda k: k[0])
      row_contours_list.append((d,row_list))
      # row_contours_list.append(row_list)
      # self.row_cords = row_contours_list
      self.results[str(page_num)] ={"rows_list": row_contours_list}

    # filename=os.path.join(self.save_path,'_page_{0}_rows.jpg'.format(page_num))
    # cv2.imwrite(filename, image)

    # print("Rows DONE",row_contours_list)

  def grouping_cols(self,page_num):
    dict_rows={}
    target_y=[]
    for contour in self.map:
      if self.map[contour][0] not in target_y:
        target_y.append(self.map[contour][0])
      if self.map[contour][0] in dict_rows:
        dict_rows[self.map[contour][0]].append(contour)
      else:
        dict_rows[self.map[contour][0]]=[contour]
    sorted_numbers = sorted(target_y)
    result=[]
    current_group = [sorted_numbers[0]]

    for i in range(1, len(sorted_numbers)):
        if abs(sorted_numbers[i] - sorted_numbers[i-1]) <= 5:
            current_group.append(sorted_numbers[i])
        else:
            result.append(current_group)
            current_group = [sorted_numbers[i]]
    result.append(current_group)
    f_contours=[]
    # print("dict_columns",dict_rows)
    for k in result:
      l=[]
      for h in k:
        for g in range(len(dict_rows[h])):
          l.append(dict_rows[h][g])
      f_contours.append(l)
    colors = np.random.randint(0,255, size=(len(self.table_contours), 4), dtype=np.uint8)
    image=cv2.imread(self.image)

    for d,f in enumerate(f_contours):
      color = (int(colors[d][0]), int(colors[d][1]), int(colors[d][2]))
      # print(d,"Columns", f)
      for val in f:
        x,y,w,h=val
        cv2.rectangle(image, (x, y), (x+w, y+h),color, 3)
    # filename=os.path.join(self.save_path,'_page_{0}_columns.jpg'.format(page_num))
    # cv2.imwrite(filename, image)
    # print("Columns DONE")



  def filterV(self,line_coordinates,maxline,page_num):
    image=cv2.imread(self.image)
    line_coordinates = [[list(a) ,list(b)]for a,b in line_coordinates]

    for line in line_coordinates:
      # print(line[0][1])
      if abs(line[0][1] - maxline[0][1]) <= 50:
        line[0][1]=maxline[0][1]
      if abs(line[1][1] - maxline[1][1]) <= 50:
        line[1][1]=maxline[1][1]

      cv2.circle(image,line[0], 5,(255,0,0), -1)
      cv2.circle(image,line[1], 5, (0, 0, 255), -1)
      cv2.putText(image, f"({line[0][0]},{line[0][1]})",(line[0][0] , line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
      cv2.putText(image, f"({line[1][0]}, {line[1][1]})", (line[1][0],line[1][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    self.b_y0=maxline[0][1]
    self.b_y1=maxline[1][1]
    # filename=os.path.join(self.save_path,'_page_{0}_filterV.jpg'.format(page_num))
    # cv2.imwrite(filename, image)

    # cv2_imshow(image)
    line_coordinates = [(tuple(a) ,tuple(b)) for a,b in line_coordinates]

    # print(line_coordinates)
    return line_coordinates



  def filterH(self,line_coordinates,maxline,page_num):
    image=cv2.imread(self.image)
    line_coordinates = [[list(a) ,list(b)]for a,b in line_coordinates]

    for line in line_coordinates:
      # print(line[0][1])
      if abs(line[0][0] - maxline[0][0]) <= 50:
        line[0][0]=maxline[0][0]
      if abs(line[1][0] - maxline[1][0]) <= 50:
        line[1][0]=maxline[1][0]

      cv2.circle(image,line[0], 5,(255,0,0), -1)
      cv2.circle(image,line[1], 5, (0, 0, 255), -1)
      cv2.putText(image, f"({line[0][0]},{line[0][1]})",(line[0][0] , line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
      cv2.putText(image, f"({line[1][0]}, {line[1][1]})", (line[1][0],line[1][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    self.b_x0=maxline[0][0]
    self.b_x1=maxline[1][0]
    # filename=os.path.join(self.save_path,'_page_{0}_filterH.jpg'.format(page_num))
    # cv2.imwrite(filename, image)
    line_coordinates = [(tuple(a) ,tuple(b)) for a,b in line_coordinates]

    # print(line_coordinates)
    return line_coordinates


  def boundary(self,x1,y1,x2,y2,page_num):
    mask=self.mimg
    cv2.rectangle(mask,(x1, y1), (x2, y2), (255,255,255), 3)
    # filename=os.path.join(self.save_path,'_page_{0}_boundary.jpg'.format(page_num))
    # cv2.imwrite(filename, mask)


  def longest_line(self,line_coordinates):
    import math
    max_length=0
    max_line = None
    for coordinates in line_coordinates:
        # Extract start and end points
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]

        # Calculate the length of the line
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length > max_length:
            max_length=length
            max_line = coordinates
    return max_line
  
  
  def main(self):
    Row_contour_dict={}
    # try:
    Row_contour_dict = Table_Drawer.Extract(self)
    
    # except Exception as e:
    #     print(str(e))
    return Row_contour_dict

    








