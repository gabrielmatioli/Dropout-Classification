import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

## Preprocessing ##

df = pd.read_csv('../data/data.csv', sep=';')
df = df.loc[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})
df = df.astype(float)

categorical = ['Marital status', 'Application mode', 'Application order',
               'Course', 'Previous qualification', 'Nacionality',
               "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation", 
               'Curricular units 1st sem (credited)',
               'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
               'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 
               'Curricular units 1st sem (without evaluations)', 
               'Curricular units 2nd sem (credited)',
               'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
               'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
               'Curricular units 2nd sem (without evaluations)']

continuous = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
              'Unemployment rate', 'Inflation rate', 'GDP']

ct = ColumnTransformer([('scaler', StandardScaler(), continuous),
                        ('encoder', OrdinalEncoder(), categorical)], remainder='passthrough')

X_transformed = pd.DataFrame(ct.fit_transform(df.drop(columns=['Target'])), columns=ct.get_feature_names_out())
y = df['Target']

sampler = SMOTE()
X_sampled, y_sampled = sampler.fit_resample(X_transformed, y)

## Model Training ##

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.15, random_state=42)

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Model Testing ##

clf_report = classification_report(y_test, y_pred)
print('CLF Report: \n')
print(clf_report)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='magma')
plt.show()