import os
import matplotlib.pyplot as plt 
import numpy as np 

from sklearn.cluster import KMeans
from sklearn.utils import shuffle



def recreate_img(image):
	#print(image.shape)
	img2 = image.reshape(image.shape[0]*image.shape[1],1)

	img1 = shuffle(image.reshape(image.shape[0]*image.shape[1],-1),random_state=0)
	#print(img1.shape[0])
	kmeans = KMeans(n_clusters=3,random_state=1).fit(img1[:3*img1.shape[0]//5])
	#print(kmeans.cluster_centers_)
	pred = kmeans.predict(img2)
	dic = {i:len(pred[pred==i]) for i in np.unique(pred)}
	maxi = 0

	mode_ind = 0
	for i in dic:
		if(dic[i]>maxi):
			maxi=dic[i]
			mode_ind = i

	#print(dic)
	mini = np.inf

	#print(np.unique(pred))

	new_im = np.zeros_like(image).astype(np.float64)

	count = 0
	for i in range(0,new_im.shape[0]):
		for j in range(0,new_im.shape[1]):
			new_im[i,j] = kmeans.cluster_centers_[pred[count]]
			count+=1

	return pred,new_im,kmeans.cluster_centers_[mode_ind],kmeans
import pickle
def get_color(main_image):

	r = main_image[:,:,0].astype(np.float64)
	g = main_image[:,:,1].astype(np.float64)
	b = main_image[:,:,2].astype(np.float64)
	
	gray = 0.3*r + 0.59*g + 0.11*b
	
	imgr = gray - r
	imgb = gray - b
	imgg = gray - g
	
	dic = {'red':imgr,'blue':imgb,'green':imgg}

	new_im = np.zeros_like(imgr,dtype=np.float64).reshape(imgr.shape[0]*imgr.shape[1],-1)
	clfs = {}
	for color in dic:
		lab,rec_img,modal_col,clf = recreate_img(dic[color])
		
		clfs[color] = clf

		rec_img_temp = rec_img.reshape(rec_img.shape[0]*rec_img.shape[1],-1)

		new_im[np.where(rec_img_temp!=modal_col)]=1.0

	new_im = new_im.reshape(imgr.shape[0],imgr.shape[1])
	
	with open('./model_.pkl','wb') as saved_clf:
		pickle.dump(clfs,saved_clf)

	#plt.imshow(new_im.reshape(imgr.shape[0],imgr.shape[1])),plt.show()


	new_im2 = np.zeros((new_im.shape[0],new_im.shape[1],3),dtype=np.float64)

	new_im2[:,:,0]=new_im
	new_im2[:,:,1]=new_im 
	new_im2[:,:,2]=new_im
	#plt.imshow(np.multiply(main_image,new_im2)),plt.show()

	# for i in range(img1.shape[0]):

	# 	new_im[i,:] = img2[i,]
import scipy.misc
def test(clf_dir,main_image):
	clf = pickle.load(open(clf_dir,'rb'))
	
	gray = 0.3*main_image[:,:,0] + 0.59*main_image[:,:,1] + 0.11*main_image[:,:,2]
	
	r = gray - main_image[:,:,0].astype(np.float64)
	g = gray - main_image[:,:,1].astype(np.float64)
	b = gray - main_image[:,:,2].astype(np.float64)
	
	im_dic = {'red':r , 'blue':b,'green':g}

	mask = np.zeros_like(gray)

	for k in clf:

		pred = clf[k].predict(im_dic[k].reshape(main_image.shape[0]*main_image.shape[1],1))
		dic = {x:len(pred[pred==x]) for x in np.unique(pred)}
		
		maxi = 0
		for i in dic:
			if(dic[i]>maxi):
				maxi=dic[i]
				max_ind = i

		pred[pred==max_ind] = 0.0
		pred[pred!=max_ind] = 1.0
		pred = pred.reshape((main_image.shape[0],main_image.shape[1]))

		mask+=pred

	mask[mask>=1]=1.0
	new_mask = np.zeros_like(main_image)

	new_mask[:,:,0]=mask
	new_mask[:,:,1]=mask
	new_mask[:,:,2] = mask
	return np.multiply(main_image,new_mask)

#for i in os.listdir('/data/gabriel/betty_mr/color/'):
import argparse

#if "__name__" == "__main__": 

parser = argparse.ArgumentParser()
parser.add_argument('--frame_dir',help='directory of frames')
parser.add_argument('--dest_dir',help='directory of annotations')    
args = parser.parse_args()
#print('here1')
for ct_imgs in os.listdir(args.frame_dir):
    img = plt.imread(args.frame_dir+'/'+ct_imgs)

    rec_im = test('./model_.pkl',img)
    #size_rec = rec_im.shape
    rec_im[:,:55,:] = 0
    rec_im[1050:,:,:] = 0
    #print('here2')
    scipy.misc.imsave(args.dest_dir+'/'+ct_imgs,rec_im)
        
        
#for j in os.listdir('/data/gabriel/betty_mr/color/'+i+'/'):

'''
img = plt.imread('/data/gabriel/betty_mr/orig_img.png')
rec_im = test('./model_.pkl',img)

scipy.misc.imsave('/data/gabriel/betty_mr/example_img.png',rec_im)
'''

#imag = plt.imread('/data/gabriel/betty_mr/G IV/66(23)42.png').astype(np.float64)
#plt.imshow(imag),plt.show()
#get_color(imag)

#test('./model_.pkl',plt.imread('/data/gabriel/betty_mr/G IV/66(72)26.png').astype(np.float64))

#exit()
#plt.imshow(imag),plt.show()
#exit()
#c_table = plt.imread('/data/gabriel/betty_mr/color_table2.png').astype(np.float64)

#c_table = c_table.reshape(c_table.shape[0]*c_table.shape[1],-1)

'''
c_table = imag[25:140,594:]
c_table=c_table.reshape(c_table.shape[0]*c_table.shape[1],-1)

print(type(c_table))
print(type(c_table[0]))
#print(c_table.shape)
#plt.imshow(c_table),plt.show()
#plt.imshow(c_table),plt.show()
#exit()
c_table2 = c_table.copy()
#print(c_table2)

new_table = np.zeros_like(c_table2)

# for i in range(0,c_table2.shape[0]):
# 	for j in range(0,c_table2.shape[1]):
# 		if(np.any(c_table==c_table2[i,j,:])):
# 			new_table[i,j,:] = c_table2[i,j,:]

# plt.imshow(new_table),plt.show()

# exit()
#c_table_new = np.array([i for i in c_table2[0] if i.sum()>0.0])

#print(c_table_new)
flag=0
new_im = np.zeros_like(imag).astype(np.float64)
for i in range(0,imag.shape[0]):
	for j in range(0,imag.shape[1]):
		#print([imag[i,j,0],imag[i,j,1],imag[i,j,2]])

		if np.any(imag[i,j,0] == c_table2[:,0]) and np.any(imag[i,j,1] == c_table2[:,1]) and np.any(imag[i,j,2] == c_table2[:,2]) :
			#print(imag[i,j])
			#flag=1
			#break

			new_im[i,j,:]=imag[i,j,:]
	# if(flag==1):
	# 	break
plt.imshow(new_im),plt.show()
'''