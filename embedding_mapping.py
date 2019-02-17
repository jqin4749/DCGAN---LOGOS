import numpy as np
import pickle

mapping={}
for i in range(1,23):
    print(str(i)+" finished")
    data=np.load("./www_embeddings/embeddings_"+str(i)+".npy")
    idx=0
    for row in data:
        app_id=row[0]
        if(app_id != None):
            app_id=app_id[:len(app_id)-4]
            mapping[app_id]={"group_nb":str(i),"index":idx}
            idx=idx+1
    print(str(i)+" finished") 
with open("./embedding_mapping.p","wb+") as f:
    pickle.dump(mapping,f,pickle.HIGHEST_PROTOCOL)
print("job finished")
