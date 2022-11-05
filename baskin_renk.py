import cv2 
from sklearn.cluster import KMeans
import numpy as np


def getdcolor(img, n):
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    return colors.astype(int)


img = cv2.imread(r'C:\Users\Yusuf\Desktop\machine_learning\rgb_color\image2.png')
assert not isinstance(img,type(None)), 'image not found'

cluster = 5 #KAÇ TANE RENK BULUNACAĞINI BELİRLEYEN DEĞERİ VERİYORUZ
colors = getdcolor(img, cluster)  

#BULUNAN RENKLERİ KULLANARAK COLORBAR OLUŞTURUYORUZ
colorbar = np.zeros((500,500,3), dtype=np.uint8)
colorbar[:,:100]= colors[0]
colorbar[:,100:200]=colors[1]
colorbar[:,100:300]=colors[2]
colorbar[:,100:400]=colors[3]
colorbar[:,400:]=colors[4]
cv2.imshow("camera",img)
cv2.imshow("Colorbar", colorbar)
cv2.waitKey()
cv2.destroyAllWindows()


#DOSYA OLUŞTURMA-YAZMA-KAYDETME İŞLEMLERİNİ YAPIYORUZ 
folder_rgb =  open(r"C:\Users\Yusuf\Desktop\machine_learning\rgb_color\rgb_outputs.txt", "w")
folder_rgb.write(str(colors[range(cluster)]))
folder_rgb.close()



