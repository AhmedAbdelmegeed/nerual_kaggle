import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
from preprocess import *
from sklearn.metrics import accuracy_score

review_col = "review_description"
rating_col = "rating"

loaded_model = load_model('best_model1.hdf5')
res = global_preprocess_sentence("train_subset.xlsx",False)


y_pred_probs = loaded_model.predict(res)
y_pred = np.argmax(y_pred_probs, axis=1)

for i in range(len(y_pred)):
    y_pred[i] -= 1


df = pd.read_excel("train_subset.xlsx")

f = [0,0,0]
for i in range(len(y_pred)):
    p = y_pred[i] + 1
    act = df["rating"][i] + 1
    if(p != act):
        f[act] = f[act] + 1


print(f)
# Evaluate the model
accuracy = accuracy_score(df['rating'], y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")

