
# coding: utf-8

# In[58]:


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


# In[59]:


present_dir=os.getcwd()
image_dir=present_dir+'\IMAGES'
#\*png'
save_dir=present_dir+'\RESULTS'
print(save_dir)


# In[60]:


import glob
images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(image_dir+r'\*png')]


# In[61]:


len(images)


# In[62]:


def prepare_answer_key(num_q):
    np.random.seed(5)
    ANSWER_KEY={}
    for i in range(num_q):
        ANSWER_KEY[i]="NA"
    ans=np.random.randint(low=0, high=4, size=num_q)
    for i in range(num_q):
        ANSWER_KEY[i]=ans[i]
    return ANSWER_KEY


# In[63]:


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


# In[64]:


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


# In[65]:


def get_master(TopBox, thresh):   
    Master=[]
    for i in range(0,120,12):
        cnts=contours.sort_contours(TopBox[i:i+12])[0]
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
    


# In[66]:


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


# In[67]:


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


# In[68]:


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


# In[69]:


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
        


# In[70]:


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


# In[77]:


def get_details(master):
    school_id=['X', 'X', 'X', 'X', 'X']
    class_id=['X', 'X']
    section_id=['X']
    roll_no=['X', 'X']
    test_id=['X', 'X']
    
    for offset in range(0,5):
        flag=False
        for i in range(0+offset, 109+offset, 12):
            print(i, master[i])
            if(master[i]==1 and flag==True):
                school_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                school_id[offset]=math.floor((i-offset)/12)
    
    for offset in range(0,2):
        flag=False
        for i in range(5+offset, 114+offset, 12):
            if(master[i]==1 and flag==True):
                roll_no[offset]='X'
                break
            if(master[i]==1):
                flag=True
                roll_no[offset]=math.floor((i-5-offset)/12)
    
    for offset in range(0,2):
        flag=False
        for i in range(7+offset, 115+offset, 12):
            if(master[i]==1 and flag==True):
                test_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                test_id[offset]=math.floor((i-7-offset)/12)
                
    for offset in range(0,2):
        flag=False
        for i in range(9+offset, 118+offset, 12):
            if(master[i]==1 and flag==True):
                class_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                class_id[offset]=math.floor((i-9-offset)/12)
                
    for offset in range(0,1):
        flag=False
        for i in range(11+offset, 120+offset, 12):
            if(master[i]==1 and flag==True):
                section_id[offset]='X'
                break
            if(master[i]==1):
                flag=True
                section_id[offset]=chr(math.floor((i-10-offset)/12)+ord('A'))
                
    return school_id, roll_no, test_id, class_id, section_id


# In[78]:


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


# In[79]:


def evaluate_image_batch(images, num_q):
    ANSWER_KEY=prepare_answer_key(num_q)
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
        
        
        
        
        if(ratio>=1.44 or len(questionCnts)!=(120+(4*num_q))):
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
        
        TopBox=AllCircles[0:120]
        questionCnts=AllCircles[120:]
        
        
        
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


# In[80]:


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

# In[81]:


mapping=list(zip(SCHOOL, CLASS, SECTION, ROLL, TEST_ID, Status, ANSWERS))


# In[82]:


import pandas as pd
df=pd.DataFrame(mapping, columns=['SCHOOL_id','CLASS_id', 'SECTION_id','ROLL', 'TEST_id', 'Status','ANSWERS'])
df.head(n=10)


# In[57]:


df.to_csv("Results.csv")


# ## Mapping explained
# mapping[i] details for ith student
# 
# mapping[i][0] is the roll number of ith student in list form
# 
# mapping[i][1] returns answer array of ith student 

# In[20]:


image = cv2.cvtColor(cv2.imread(r"C:\Users\Gautam\Desktop\OMR\IMAGES\20190622193251_orig.png"), cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged=edges_det(image, 200, 250)


# In[21]:


cv2.imwrite("image.jpg", image)
cv2.imwrite("gray.jpg", gray)
cv2.imwrite("edged.jpg", edged)
edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 11))) ## closing operation after finding edges close the gaps


# In[22]:


cv2.imwrite("edgedn.jpg", edged)


# In[ ]:


conts=find_page_contours(edged, resize(image))
implt(image)
image.shape


# In[ ]:


contoured=cv2.drawContours((image.copy()), [conts], -1, (0, 255, 0), 3)


# In[ ]:


paper=(cv2.rotate(image,rotateCode=cv2.ROTATE_90_CLOCKWISE))
warped=(cv2.rotate(gray,rotateCode=cv2.ROTATE_90_CLOCKWISE))


# In[ ]:


cv2.imwrite("11contouredn.jpg", contoured)


# In[ ]:


implt(paper)


# In[ ]:


conts


# In[ ]:


conts=conts.dot(ratio(image))


# In[ ]:


cv2.imwrite("5contoured.jpg", contoured)


# In[ ]:


##paper = persp_transform(image, conts)
##warped = persp_transform(gray, conts)


# In[ ]:


##paper=image
##warped=gray


# ## EXPERIMENTAL LINE

# In[ ]:


implt(paper)


# In[ ]:


##  cnts = cv2.findContours(warped.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
## cv2.drawContours(warped, cnts, -1, (0,255,0), 3)


# In[ ]:


cv2.imwrite("6paper.jpg", paper)
cv2.imwrite("7warped.jpg", warped)
##warped=resize(warped)
##paper=resize(paper)


# In[ ]:


implt(warped)


# ratio=(warped.shape[1]/warped.shape[0])
# ratio=np.int0(ratio)
# upsampled=np.ones(warped.shape)
# upsampled=cv2.pyrUp(warped, upsampled)

# upsampled.shape
# implt(upsampled)

# In[ ]:


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


# In[ ]:


warped=remove_shadows(warped)
implt(warped)


# In[ ]:


warpedcopy=cv2.resize(warped, (430,600))
papercopy=cv2.resize(paper, (430,600))


# In[ ]:


thresh = cv2.threshold(warpedcopy, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


# In[ ]:


## thresherode=cv2.erode(thresh, np.ones((3,3), np.uint8))
## threshopen=cv2.dilate(thresherode, np.ones((3,3), np.uint8))


# In[ ]:


print(thresh.shape[1])
thresh.shape[0]


# In[ ]:


cv2.imwrite("13eroded.jpg", thresherode)


# ratio=(thresh.shape[1]/thresh.shape[0])
# upsampled=np.ones(thresh.shape)
# upsampled=cv2.pyrUp(thresh, upsampled, (800, 594))

# upsampled.shape

# implt(upsampled)

# In[ ]:


cv2.imwrite("8thresh.jpg", thresh)


# In[ ]:


implt(thresh)


# In[ ]:


def find_questions(thresh, paper):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    tester=cv2.drawContours(paper.copy(), cnts, -1, (255,0,0), 2)

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        area = cv2.contourArea(c)
        print(w, h, ar, len(approx), area)
        if ((w >= 12 and h >=12) and (w<=16 and h<=16) and (area>90 and area<200) and len(approx)>=8 and ar >= 0.8 and ar <= 1.4):
            questionCnts.append(c)
    return questionCnts


# In[ ]:


test_conts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
test_conts=imutils.grab_contours(test_conts)
tester=cv2.drawContours(paper.copy(), test_conts, -1, (255,0,0), 2)
cv2.imwrite("tester.jpg", tester)
AllTests=imutils.contours.sort_contours(test_conts, method="top-to-bottom")[0]
xyz=[]
for c in AllTests:    
    (x, y, w, h) = cv2.boundingRect(c)
    approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    area = cv2.contourArea(c)
    print(w,h,w/float(h),len(approx), area)
        


# In[ ]:


questionCnts=find_questions(thresh, paper)
AllCircles = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

maxm = []
##ans_array=[]
##roll=[]


# In[ ]:


tester2=cv2.drawContours(paper.copy(), xyz, -1, (255,0,0), 2)
cv2.imwrite("test2.jpg", tester2)


# In[ ]:


ROLL_NO=AllCircles[0:80]
questionCnts=AllCircles[80:]


# In[ ]:


questionCnts=imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]


# In[ ]:


num_q=30


# In[ ]:


print(len(ROLL_NO))
print(len(questionCnts))
if(len(ROLL_NO)!=80):
    raise Exception("All Roll numbers not recognised, please take pic again")
if(len(questionCnts)!=(num_q*4)):
    raise Exception("All questions not recognised, please take pic again")


# In[ ]:


cv2.drawContours(cv2.resize(paper, (430,600)), ROLL_NO, -1, (255,0,100), 3)
cv2.imwrite("12rolls.jpg", paper)


# ## DEBUGGING CELL

# 
# 
# 
# for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
# 	count+=1
# 	# sort the contours for the current question from
# 	# left to right, then initialize the index of the
# 	# bubbled answer
# 	cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
# 	bubbled = None
# 	flag=False
# 	# loop over the sorted contours
# 	for (j, c) in enumerate(cnts):
# 		# construct a mask that reveals only the current
# 		# "bubble" for the question
# 		mask = np.zeros(thresh.shape, dtype="uint8")
# 		cv2.drawContours(mask, [c], -1, 255, -1)
# 
# 		# apply the mask to the thresholded image, then
# 		# count the number of non-zero pixels in the
# 		# bubble area
# 		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
# 		total = cv2.countNonZero(mask)
# 
# 		minm.append(total)
#           
# 		# if the current total has a larger number of total
# 		# non-zero pixels, then we are examining the currently
# 		# bubbled-in answer
# 		if bubbled is None or total > bubbled[0]:
# 			bubbled = (total, j) # What ???
# 	maxm.append(bubbled[0])
# 	# initialize the contour color and the index of the
# 	# *correct* answer
# 	color = (0, 0, 255)
# 	k = ANSWER_KEY[q]
# 	if k == bubbled[1]:
# 			color = (0, 255, 0)
# 			correct += 1        
# 	# check to see if the bubbled answer is correct 
# 	# draw the outline of the correct answer on the test
# 	cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# print(minm)
# for o in minm:
#     maxm.remove(o)
# print(maxm)
# 

# In[ ]:


minm=[]
maxm=[]


# In[ ]:


def get_answers(questionsCnts, thresh, ANSWER_KEY, ans_array):
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        

        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        bubbled = None
        flag=False

        for (j, c) in enumerate(cnts):

            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)


            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            minm.append(total)
            if (flag and total>700): ##was 220
                bubbled=None
                break
            if(total>700):
                flag=True

            if(total>700):
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

        cv2.drawContours(papercopy, [cnts[k]], -1, color, 3)
    return ans_array


# In[ ]:


def get_roll_number(ROLL_NO, thresh):
    roll=[]
    count=-1;
    idx=0
    for i in range(0,80, 10):
        count+=1;
        cnts = contours.sort_contours(ROLL_NO[i:i + 10])[0]
        bubbled = None
        flag=False
        for (j, c) in enumerate(cnts):

            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            maxm.append(total)
            if (flag and total>700):
                bubbled=None
                break
            if(total>700):
                flag=True
            if(total>700):
                bubbled=(total,j)
            color = (0, 255, 255)
        if(bubbled):
            roll.append(bubbled[1])
        else:
            roll.append("X")
         
    return roll


# In[ ]:


questions=cv2.drawContours(warpedcopy.copy(), questionCnts, -1, (0,255,0), 3)


# In[ ]:


cv2.imwrite("9questions.jpg", questions)


# In[ ]:


implt(questions)


# In[ ]:



ans_array=[]
ans_array=get_answers(questionCnts, thresh, ANSWER_KEY, ans_array)
roll=get_roll_number(ROLL_NO, thresh)
print(ans_array)
print(len(ans_array))
print(roll)


# In[ ]:


score = (correct / len(questionCnts)) * 100
print(correct)
print("[INFO] Score: {:.0f}%".format(score))
cv2.putText(paper, "{:.0f}%".format(score), (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Result", papercopy)
cv2.imwrite("Result.jpg", papercopy)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[ ]:


def write_to_csv(ans_array, filename):
    file=open(filename, "w+")
    for x in ans_array:
        file.write(str(x))
        file.write(',')
    file.close()    


# In[ ]:


write_to_csv(ans_array, "abc.txt")

