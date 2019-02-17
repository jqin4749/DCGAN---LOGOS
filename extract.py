import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import shutil
import PIL
from PIL import Image
import psutil

def knn(data,k,target,a):
    values = list(data.values())
    keys = list(data.keys())

    neigh = NearestNeighbors(n_neighbors=k,metric='cosine')
    neigh.fit(values)
    dist, ind=neigh.kneighbors([target])
    found_nbs={}
    if a==0:
        for i in ind[0,:]:
            found_nbs[keys[i]]= target
        return found_nbs
    else:
        for i in ind[0,:]:
            found_nbs[keys[i]+'*'+str(a)]= values[i]
        return found_nbs

def printf(p):
    with open('out.o','a+') as f:
        print(p,file=f)


def print_mem_info():
    process = psutil.Process(os.getpid())
    printf("current memory usage:"+str(process.memory_info().rss/float(2**30)))  # in GB 


group_name=["506"]
keyname=["com.ea.games.r3_row"]

printf("about to read mapping")
em_mapping=np.load("embedding_mapping.p")
printf("mapping reading finished")

result={}
result_img={}
for i,j in zip(group_name,keyname):
    filename="Text_Embedding/embedding_100_200_"+i+".p"
    data=np.load(filename)
    
    gn=em_mapping[j]['group_nb']
    idx=em_mapping[j]['index']
    data_img=np.load("./www_embeddings/embeddings_"+gn+".npy")
    result_img[j+"*"+i]=np.copy(data_img[idx][1:4097])
    result[j+"*"+i]=data[j]
del data_img

printf("first stage finished")

with open('picked_embed_v2.p','wb+') as f:
    pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)


final_nearest_combined={}
final_nearest={}
for j,base_img in zip(group_name, keyname):
    dic_res={}
    for i in range(1,1002):
        filename="Text_Embedding/embedding_100_200_"+str(i)+".p"
        data=np.load(filename)
        found_nbs=knn(data,500,result[base_img+'*'+j],i)
        dic_res={**found_nbs,**dic_res}
    
    dic_res=knn(dic_res,10000,result[base_img+'*'+j],0)
    values=list(dic_res.values())
    keys=list(dic_res.keys())

    
    printf("start loading image embeddings")
    print_mem_info() 
    f_im_ems={}
    for gn in range(1,23):
        print_mem_info()
        im_ems=np.load("./www_embeddings/embeddings_"+str(gn)+".npy")
        printf("loading image embeddings:"+str(gn))
        print_mem_info()
        for i in keys:
            filename=i.split('*')[0]
            group_nb=em_mapping[filename]['group_nb']
            idx=em_mapping[filename]['index']
            if(int(group_nb)==gn):
              #  printf("matching success "+str(gn)+' '+i)
                f_im_ems[i]=np.copy(im_ems[idx][1:4097])

    printf("finish loading image embeddings")
    del im_ems
    found_nbs_img=knn(f_im_ems,5000,result_img[base_img+'*'+j],0)
    del f_im_ems
    for i in list(found_nbs_img.keys()):
        final_nearest[i]=dic_res[i]
    final_nearest_combined={**final_nearest,**final_nearest_combined}

final_nearest_combined={**result,**final_nearest_combined}

# keys=list(final_nearest_combined.keys())
# for i in keys:
#     filename=i.split('*')[0]
#     groupnumber=i.split('*')[1]
#     shutil.copy("./images/"+groupnumber+"/"+filename+".jpg","./selected/")


with open('final_nearest_combined_v2.p','wb+') as f:
    pickle.dump(final_nearest_combined,f,pickle.HIGHEST_PROTOCOL)
# print(len(list(final_nearest.keys())))



data=final_nearest_combined
keys=list(data.keys())
basewidth = 128
for i in keys:
    filename=i.split('*')[0]
    groupnumber=i.split('*')[1]
    img=Image.open("./images/"+groupnumber+"/"+filename+".jpg")
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')
    img.save("./selected/"+filename+".jpg")
printf("resizing finished")

 #   shutil.copy("./images/"+groupnumber+"/"+filename+".jpg","./selected/")


