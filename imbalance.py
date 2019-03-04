from imblearn.over_sampling import SMOTENC
sm = SMOTENC(random_state=42, categorical_features=CATEGORICAL_FEATURES_INDEX)
