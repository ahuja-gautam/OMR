
# coding: utf-8

# In[33]:


from imutils import contours
import numpy as np
import imutils
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import csv
import os
import math


# In[34]:


present_dir=os.getcwd()
image_dir=present_dir+'\IMAGES'
#\*png'
save_dir=present_dir+'\RESULTS'
print(save_dir)


# In[35]:


import glob
images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(image_dir+r'\*png')]


# In[36]:


len(images)


# In[37]:


def prepare_answer_key(num_q):
    np.random.seed(5)
    ANSWER_KEY={}
    for i in range(num_q):
        ANSWER_KEY[i]="NA"
    ans=np.random.randint(low=0, high=4, size=num_q)
    for i in range(num_q):
        ANSWER_KEY[i]=ans[i]
    return ANSWER_KEY


# In[38]:


def remove_shadows(gray_scale):
    rgb_planes=cv2.split(gray_scale)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img=np.ones(shape=diff_img.shape)
        norm_img=cv2.normalize(diff_img,norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    return cv2.merge(result_norm_planes)


# In[39]:


def get_answers(questionCnts, thresh, ANSWER_KEY,paper):
    ans_array=[]
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        

        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        bubbled = None
        flag=False

        for (j, c) in enumerate(cnts):

            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)


            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if (flag and total>135): ##eariler 135
                bubbled=None
                break
            if(total>135):
                flag=True

            if(total>135):
                bubbled=(total,j)

        color = (0, 0, 255)
        k = ANSWER_KEY[q]

        if bubbled and k == bubbled[1]:
            color = (0, 255, 0)
            
            ans_array.append(k)       
        elif not bubbled:
            ans_array.append("NA")
        elif bubbled and k!=bubbled[1]:
            ans_array.append(bubbled[1])  

        cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    return ans_array


# In[40]:


def get_master(TopBox, thresh):   
    Master=[]
    for i in range(0,130,13):
        cnts=contours.sort_contours(TopBox[i:i+13])[0]
        for(j, c) in enumerate(cnts):            
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            if(total>135):
                Master.append(1)
            else:
                Master.append(0)
    return Master
    


# In[41]:


def get_roll_number(ROLL_NO, thresh):
    roll=[]
    count=-1;
    idx=0
    X_encounter=False
    for i in range(0,120, 12):
        count+=1;
        cnts = contours.sort_contours(ROLL_NO[i:i + 10])[0]
        bubbled = None
        flag=False
        for (j, c) in enumerate(cnts):

            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if (flag and total>135): ##135
                bubbled=None
                break
            if(total>135):
                flag=True
            if(total>135):
                bubbled=(total,j)
            color = (0, 255, 255)
        if(bubbled):
            roll.append(str(bubbled[1]))
        else:
            ##roll.append("X")
            ##X_encounter=True
            continue
         
    return roll, X_encounter


# In[42]:


def find_questions(thresh, paper):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        area = cv2.contourArea(c) ##earlier 13 and 17
        if ((w >= 47 and h >=47) and (w<=61 and h<=61) and len(approx)>=8 and ar >= 0.75 and ar <= 1.2): ##make it 18 22 when circles have been resized
            questionCnts.append(c)
    return questionCnts


# In[43]:


##below function for precision
def find_questions_precise(thresh, paper):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    ## tester=cv2.drawContours(paper.copy(), cnts, -1, (255,0,0), 2)

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        area = cv2.contourArea(c)
        ## print(w, h, ar, len(approx), area)
        if ((w >= 13 and h >=13) and (w<=17 and h<=17) and len(approx)>=8 and (area>110 and area<185) and ar >= 0.75 and ar <=1.33):
            questionCnts.append(c)
    return questionCnts


# In[44]:


def shuffle_answers(answer, num_q):
    temp=[]
    if(num_q==30):
        for i in range(0,len(answer),2):
            temp.append(answer[i])
        for i in range(1,len(answer),2):
            temp.append(answer[i])
    else:
        for i in range(0,len(answer),3):
            temp.append(answer[i])
        for i in range(1,len(answer),3):
            temp.append(answer[i])
        for i in range(2,len(answer),3):
            temp.append(answer[i])
    return temp   
        


# In[45]:


def find_edges(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    ok=False

# ensure that at least one contour was found
    if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
 
	# loop over the sorted contours
        for c in cnts:
		# approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,  0.02* peri, True)

     
		# if our approximated contour has four points,
		# then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx            
                ok=True
                break
    return docCnt


# In[46]:


def get_details(master):
    school_id=['X', 'X', 'X', 'X', 'X']
    class_id=['X', 'X']
    section_id=['X', 'X']
    roll_no=['X', 'X']
    test_id=['X', 'X']
    
    step=13
    for offset in range(0,5):
        flag=False
        for i in range(0+offset, 118+offset, step):
            print(i, master[i])
            if(master[i]==1 and flag==True):
                school_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                school_id[offset]=math.floor((i-offset)/step)
    
    for offset in range(0,2):
        flag=False
        for i in range(5+offset, 123+offset, step):
            if(master[i]==1 and flag==True):
                class_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                class_id[offset]=math.floor((i-5-offset)/step)
    
    for offset in range(0,2):
        flag=False
        for i in range(7+offset, 125+offset, step):
            if(master[i]==1 and flag==True):
                section_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                section_id[offset]=math.floor((i-7-offset)/step)
                
    for offset in range(0,2):
        flag=False
        for i in range(9+offset, 127+offset, step):
            if(master[i]==1 and flag==True):
                roll_no[offset]='X'
                break
            if(master[i]==1):
                flag=True
                roll_no[offset]=math.floor((i-9-offset)/step)
                
    for offset in range(0,2):
        flag=False
        for i in range(11+offset, 129+offset, step):
            if(master[i]==1 and flag==True):
                test_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                test_id[offset]=math.floor((i-11-offset)/step)
                
    return school_id, roll_no, test_id, class_id, section_id


# In[47]:


def checkX(school_id, class_id, section_id, roll_no, test_id):
    for x in school_id:
        if (x=='X'):
            return True
    for x in class_id:
        if(x=='X'):
            return True
    for x in section_id:
        if(x=='X'):
            return True
    for x in roll_no:
        if(x=='X'):
            return True
    for x in test_id:
        if(x=='X'):
            return True
    return False


# In[48]:


def evaluate_image_batch(images, num_q):
    ANSWER_KEY=prepare_answer_key(num_q) ##commment this line out when taking input as custom answer key
    SCHOOL_id=[]
    CLASS_id=[]
    SECTION_id=[]
    ROLL_no=[]
    TEST_id=[]
    Status=[]
    Answers=[]
    d=-1
    for image in images:
        d+=1
        X_encountered=False
        ## images have to be rotated 90 degrees first
        
        
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ##score = cv2.Laplacian(image, cv2.CV_64F).var()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        docCnt=find_edges(edged)
        paper = four_point_transform(image, docCnt.reshape(4, 2))
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
        print(warped.shape[0]/warped.shape[1], d)
        ratio=(warped.shape[0]/warped.shape[1])
        paper=cv2.resize(paper, (430,600))
        warped=cv2.resize(paper, (430,600))
        warped=cv2.cvtColor(paper,cv2.COLOR_BGR2GRAY)
        warped=remove_shadows(warped)
    
        
# ensure that at least one contour was found            

        
        thresh = cv2.threshold(warped.copy(), 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        questionCnts=find_questions_precise(thresh,paper)
        
        
        
        
        if(ratio>=1.44 or len(questionCnts)!=(130+(4*num_q))):
            SCHOOL_id.append("Flagged")
            CLASS_id.append("Flagged")
            SECTION_id.append("Flagged")
            ROLL_no.append("Flagged")
            TEST_id.append("Flagged")
            Answers.append("Flagged")
            cv2.drawContours(paper, questionCnts, -1, (255,0,100), 3)
            cv2.imwrite(save_dir+r"\%dquestion_not_detected.jpg"%d, paper)
            Status.append("Question Detection/Page Detection Error")            
            continue
            
            
        AllCircles = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        
        TopBox=AllCircles[0:130]
        questionCnts=AllCircles[130:]
        
        
        
        Master=get_master(TopBox, thresh)
        
        
        '''
        if(len(TopBox)!=120 or len(questionCnts)!=(4*num_q)):
            print(d)
            SCHOOL_id.append("Flagged")
            CLASS_id.append("Flagged")
            SECTION_id.append("Flagged")
            ROLL_no.append("Flagged")
            TEST_id.append("Flagged")
            Answers.append("Flagged")
            cv2.drawContours(paper, questionCnts, -1, (255,0,100), 3)
            cv2.imwrite(save_dir+r"\%dquestion_not_detected.jpg"%d, paper)
            Status.append("Question Detection/Page Detection Error")
            continue
            
         '''   
            
        cv2.drawContours(paper, TopBox, -1, (255,0,100), 3)
        try:
            answers=get_answers(questionCnts, thresh, ANSWER_KEY, paper)
        except:
            print(d, "***")
            SCHOOL_id.append("Flagged")
            CLASS_id.append("Flagged")
            SECTION_id.append("Flagged")
            ROLL_no.append("Flagged")
            TEST_id.append("Flagged")
            Answers.append("Flagged")
            cv2.drawContours(paper, questionCnts, -1, (255,0,100), 3)
            cv2.imwrite(save_dir+r"\%dquestion_not_detected.jpg"%d, paper)
            Status.append("Question Detection/Page Detection Error")            
            continue
            
            
        
        school_id, roll_no, test_id, class_id, section_id=get_details(Master)
        print(school_id, roll_no, test_id, class_id, section_id)    
        '''
        except:
            print(d, "****")
            SCHOOL_id.append("Flagged")
            CLASS_id.append("Flagged")
            SECTION_id.append("Flagged")
            ROLL_no.append("Flagged")
            TEST_id.append("Flagged")
            Answers.append("Flagged")
            cv2.drawContours(paper, questionCnts, -1, (255,0,100), 3)
            cv2.imwrite(save_dir+r"\%dquestion_not_detected.jpg"%d, paper)
            Status.append("Question Detection/Page Detection Error")
            continue
        '''
            
            
            
        SCHOOL_id.append(school_id)
        CLASS_id.append(class_id)
        SECTION_id.append(section_id)
        ROLL_no.append(roll_no)
        TEST_id.append(test_id)
        answers=shuffle_answers(answers, num_q)
        Answers.append(answers) 
        
        
        X_encountered=checkX(school_id, class_id, section_id, roll_no, test_id)
        
        if(X_encountered):
            Status.append("Check X")
        else:
            Status.append("Success")
        cv2.imwrite(save_dir+r"\%dresult.jpg"%d, paper) 
        
    return SCHOOL_id, CLASS_id, SECTION_id, ROLL_no, TEST_id, Status, Answers


# In[49]:


SCHOOL, CLASS, SECTION, ROLL, TEST_ID, Status, ANSWERS=evaluate_image_batch(images, 50)


# 
#         print(d, "out of" ,len(images), "processed")
#         ans_array=[]
#         ## image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         ## gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
#         edged=edges_det(image,200,250)
#         edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 11)))
#         conts=find_page_contours(edged, resize(image))
#         contoured=cv2.drawContours(resize(image.copy()), [conts], -1, (0, 255, 0), 3)
#         conts=conts.dot(ratio(image))
#         if(conts[0][0]==0):
#             Roll_numbers.append("Flagged")
#             Answers.append("Flagged")
#             cv2.imwrite(r"C:\Users\Gautam\Desktop\OMR\RESULTS\%dpagenotdetected.jpg"%d, contoured)
#             print(d, "*")
#             Status.append("Page_Detect_Error")
#             continue
#         paper = persp_transform(image, conts)
#         warped = persp_transform(gray, conts)
#         warped=resize(warped)
#         paper=resize(paper)

# In[50]:


mapping=list(zip(SCHOOL, CLASS, SECTION, ROLL, TEST_ID, Status, ANSWERS))


# In[51]:


import pandas as pd
df=pd.DataFrame(mapping, columns=['SCHOOL_id','CLASS_id', 'SECTION_id','ROLL', 'TEST_id', 'Status','ANSWERS'])
df.head(n=10)


# In[52]:


df.to_csv("Results.csv")


# In[ ]:

