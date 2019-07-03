
# coding: utf-8

# In[76]:


from pyzbar.pyzbar import decode
from PIL import Image


# In[77]:


def get_QR_data(image_path):
    """
    param image_path: Input the path of the image in the directory
    
    returns: List of encoding data
    """
    image=Image.open(image_path)
    image=image.transpose(Image.ROTATE_90)
    data=decode(image)
    data_string=str(data[0][0], 'utf-8')
    return data_string.split(":")


# In[78]:


## path in the form of r"C:\Users\Gautam\Desktop\OMR\IMAGES\20190627151101_orig.png" (example)

