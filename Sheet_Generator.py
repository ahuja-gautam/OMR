
# coding: utf-8

# In[1]:


from pyqrcode import QRCode
import png
import os
import cv2
from fpdf import FPDF
pdf=FPDF()
import img2pdf


# In[36]:


##d=omr id
def create_sheet(num_q, test_id,school_id, class_id, section_id, test_name, school_name, d):
    if(len(test_name)>22):
        test_name=test_name[0:22]
    if(len(school_name)>22):
        school_name=school_name[0:22]
     ## d to indicate name of sheet to overcome overwriting issues
    sheet=cv2.imread('Sheet%d.png'%num_q)
    encoding=(str(d)+':'+test_id+':'+school_id+':'+class_id+':'+section_id+':'+str(num_q)).upper()
    qr_code=pyqrcode.create(content=encoding, error='L',version=3)
    qr_code.png(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d, scale=15)
    QR=cv2.imread(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d)
    y=QR.shape[0]
    x=QR.shape[1]
    font=cv2.FONT_HERSHEY_DUPLEX 
    vert_offset=1700 ##along x axis
    hor_offset=450 ##along y axis
    sheet[hor_offset:hor_offset+x,vert_offset:vert_offset+y,]=QR
    cv2.imwrite(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d, sheet)
    copy=cv2.imread(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d)

    base=1050
    thickness=3
    size=1.5
    color=(0,0,0) ##black text
    cv2.putText(copy, 'Test:%s'%test_name, (vert_offset-100, base),font, size,color,thickness,cv2.LINE_AA)
    cv2.putText(copy, 'School:%s'%school_name, (vert_offset-100, base+60),font, size,color,thickness,cv2.LINE_AA)
    cv2.putText(copy, 'Class:%s'%class_id, (vert_offset-100, base+120),font, size, color, thickness,cv2.LINE_AA)
    cv2.putText(copy, 'Section:%s'%section_id, (vert_offset-100, base+180),font, size, color ,thickness,cv2.LINE_AA)
    cv2.imwrite(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d, copy)
    with open(r"C:\Users\Gautam\Desktop\OMR\PDFS\%d.pdf"%d, "wb") as f:
        f.write(img2pdf.convert(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d))
    os.remove(r"C:\Users\Gautam\Desktop\OMR\TEMP\%dencoding.png"%d)

# In[37]:


## create_sheet(num_q=50,test_id="T12Weekly",school_id="123456789987",class_id="123456789987",section_id="123456789987", test_name='UNNAYAN TEST SERIES12', school_name="UNNAYAN SCHOOL 1", ,d=123)

