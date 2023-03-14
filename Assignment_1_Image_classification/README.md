## image classification 作業
#### image classification pipeline
1. 將images分成train、validation、test三份dataset，train部分因時間關係，每個類別只取120張來做訓練
2. 讀取三份dataset各自用HOG做image feature extraction並把feature存成np.array
3. 分別使用K-NN、Decision Tree、Logistic Regression、SVC、adaBoost做分類
