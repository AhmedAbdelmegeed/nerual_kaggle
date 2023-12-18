import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
from preprocess import *

review_col = "review_description"
rating_col = "rating"

loaded_model = load_model('my_model.h5')
res = global_preprocess_sentence("test_subset.xlsx",False)


predictions = loaded_model.predict(res)
predictions = np.around(predictions, decimals=0).argmax(axis=1)
print(list(predictions))

df = pd.read_excel("test_subset.xlsx")

c = 0
f = 0

for i in range(len(predictions)) :
    ii = int(df[rating_col][i])
    ii += 1
    if int(predictions[i]) != ii:
        f += 1
    else  :
        c += 1 


print(c,f , f"Accuraccy : {(c*100)/(c+f)}")

