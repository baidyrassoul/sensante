# notebooks/train_model.py
# Lab 2 - Entraîner et Sérialiser un Modèle
# Projet SénSanté

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================
# ETAPE 2 : Charger et préparer les données
# =============================================

df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir X (features) et y (cible)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")


# =============================================
# ETAPE 3 : Séparer train / test
# =============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")


# =============================================
# ETAPE 4 : Entraîner le modèle
# =============================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")


# =============================================
# ETAPE 5 : Evaluer le modèle
# =============================================

y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(f"\n{comparison}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion + rapport
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualisation (optionnel)
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")


# =============================================
# ETAPE 6 : Sérialiser le modèle
# =============================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")
print(f"\nModele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder aussi les encodeurs et les feature_cols
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes.")


# =============================================
# ETAPE 7 : Tester le modèle sérialisé
# =============================================

# Recharger depuis le fichier
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"\nModele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Simuler un nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder les catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Construire le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")

print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")


# =============================================
# EXERCICE 1 : Importance des features
# =============================================

print("\n--- Importance des features ---")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")


# =============================================
# EXERCICE 2 : Tester 3 patients fictifs
# =============================================

patients_fictifs = [
    # Jeune sans symptômes
    {'age': 19, 'sexe': 'M', 'temperature': 37.0,
     'tension_sys': 120, 'toux': False, 'fatigue': False,
     'maux_tete': False, 'region': 'Dakar'},

    # Adulte avec forte fièvre
    {'age': 35, 'sexe': 'F', 'temperature': 40.2,
     'tension_sys': 115, 'toux': True, 'fatigue': True,
     'maux_tete': True, 'region': 'Thiès'},

    # Patient âgé avec toux
    {'age': 65, 'sexe': 'M', 'temperature': 38.5,
     'tension_sys': 140, 'toux': True, 'fatigue': True,
     'maux_tete': False, 'region': 'Saint-Louis'},
]

print("\n--- Test sur 3 patients fictifs ---")
for i, p in enumerate(patients_fictifs):
    s_enc = le_sexe_loaded.transform([p['sexe']])[0]
    r_enc = le_region_loaded.transform([p['region']])[0]

    feats = [
        p['age'], s_enc, p['temperature'], p['tension_sys'],
        int(p['toux']), int(p['fatigue']), int(p['maux_tete']), r_enc
    ]

    diag = model_loaded.predict([feats])[0]
    prob = model_loaded.predict_proba([feats])[0].max()

    print(f"Patient {i+1} ({p['sexe']}, {p['age']} ans, {p['region']}) "
          f"=> {diag} ({prob:.1%})")