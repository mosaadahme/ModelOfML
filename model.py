from unicodedata import name
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import classification_report
# ========= Read data ==========
data = pd.read_csv(r"C:\Users\dell\Downloads\Telegram Desktop\Crops-to-Grow-mainABDOFinal\Crops-to-Grow-mainABDO\data\Crop_recommendation.csv")

#========== Label Encoding ===========
label_encoder = LabelEncoder()
# data['label'] = label_encoder.fit_transform(data['label'])


# ========== Features selection ============
x = data.drop(['label'] , axis=1)
y = data['label']


# =====  Split data to train and test data =======
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

# =========== Scaling =============
# Train data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train , columns=x.columns)

# test data
x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test , columns= x.columns)

# =========== Build Random forest model =========
# 1- Train model
param_grid = {
'n_estimators': [50, 75,100, 150, 200,300],
}
rcv = RandomizedSearchCV(RandomForestClassifier(random_state=0),param_grid,cv=5)
rcv.fit(x_train,y_train)

# Test================================
# y_pred = rcv.predict(x_test)
# print(y_pred)
# print(classification_report(y_test,y_pred))
# 2- create pickle file
pickle.dump(rcv , open('model.pkl' , 'wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(label_encoder, open('label_encoder.pkl','wb'))
#============             Recommendation functio =============
names = data['label'].unique()
def recommend(X):
    probability=rcv.predict_proba(X)
    probability = sorted( [(x,i) for (i,x) in enumerate(probability[0])], reverse=True)
    for i,j in probability[:3]:
        print(names[j])
recommend(x_test.sample(1))
print(rcv.predict(x_test.sample()))



