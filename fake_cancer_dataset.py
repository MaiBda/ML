import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

def determine_cancer_type(age, gender):
    if age < 50 and gender == 'Male':
        return random.choice(["Prostate", "Testicular"])
    elif age < 50 and gender == 'Female':
        return random.choice(["Breast", "Cervical"])
    elif gender == "Female" and age >= 50:
        return random.choice(["Breast", "Lung"])
    elif age >= 50 and gender == 'Male':
        return random.choice(["Prostate", "Lung"])


def determine_cancer_stage(age, cancer_type):
    if cancer_type == 'Prostate' or cancer_type == 'Testicular' or cancer_type == 'Colon' :
        if age < 50:
            return random.choice(['Stage 1', 'Stage 2'])
        else:
            return random.choice(['Stage 1', 'Stage 2', 'Stage 3'])
    elif cancer_type == 'Breast' or cancer_type == 'Cervical' or cancer_type == 'Colon':
        if age < 50:
            return random.choice(['Stage 1', 'Stage 2'])
        else:
            return random.choice(['Stage 1', 'Stage 2', 'Stage 3'])
    elif cancer_type == 'Lung':
        return random.choice(['Stage 3', 'Stage 4'])
    else:
        return random.choice(['Stage 1','Stage 2', 'Stage 3', 'Stage 4'])
def determine_cancer_treatment(stage, cancer_type):
    if cancer_type == 'Prostate' or cancer_type == 'Testicular' and stage == 'Stage 1' or stage == 'Stage 2':
        return random.choice(['Surgery', 'Radiation'])
    elif cancer_type == 'Breast' or cancer_type == 'Cervical' and stage == 'Stage 1' or stage == 'Stage 2':
        return random.choice(['Surgery', 'Radiation'])
    elif cancer_type == 'Lung' and stage == 'Stage 3' or stage == 'Stage 4':
        return random.choice(['Chemotherapy' ,'Combined therapy'])
    else:
        return random.choice(['Surgery', 'Radiation','Combined therapy','Chemotherapy'])
#
def determine_outcome(stage, treatment):
    if stage == 'Stage 1' or stage == 'Stage 2' and treatment == 'Surgery':
        return random.choice(['Stable', 'Cured'])
    elif stage == 'Stage 1' or stage == 'Stage 2' and treatment == 'Radiation':
        return random.choice(['Stable', 'Cured', 'Partial response'])
    elif stage == 'Stage 3' or stage == 'Stage 4' and treatment == 'Combined therapy' or treatment == 'Chemotherapy':
        return random.choice(['Stable','Partial response','Deteriorating'])
    else:
        return random.choice(['Stable', 'Cured', 'Deteriorating','Partial response'])

def generate_fake_dataset(size):
    age = [random.randint(20, 90) for i in range(size)]
    gender = [random.choice(['Male', 'Female']) for i in range(size)]
    cancer_type = [determine_cancer_type(a, g) for a, g in zip(age, gender)]
    cancer_stage = [determine_cancer_stage(a, ct) for a, ct in zip(age, cancer_type)]
    treatment = [determine_cancer_treatment(cs, ct) for cs, ct in zip(cancer_stage, cancer_type)]
    outcome = [determine_outcome(cs, t) for cs, t in zip(cancer_stage, treatment)]
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Cancer Type': cancer_type,
        'Cancer Stage': cancer_stage,
        'Treatment': treatment,
        'Outcome': outcome
    })
    return df

df = generate_fake_dataset(size=100000)
Cancer_df = df.to_csv('Fake_cancer_DS.csv')

# df = pd.read_csv('Fake_cancer_DS.csv')
# Encode categorical features
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
df[categorical_columns] = df[categorical_columns].apply(lambda col: LabelEncoder().fit_transform(col))

# Data split into training and testing sets
X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier
model_RF = RandomForestClassifier()
model_RF.fit(X_train, y_train)

# Evaluate the model_RF on the test data
y_pred_RF = model_RF.predict(X_test)
precision = precision_score(y_test, y_pred_RF, average='macro')
recall = recall_score(y_test, y_pred_RF, average='macro')
f1 = f1_score(y_test, y_pred_RF, average='macro')
print('Accuracy:', accuracy_score(y_test, y_pred_RF))
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)