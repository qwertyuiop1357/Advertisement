class FFMFormatPandas:
    def __init__(self, CATEGORICAL_FEATURES):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None
        self.CATEGORICAL_FEATURES = CATEGORICAL_FEATURES

    def fit(self, df, CATEGORICAL_FEATURES, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            if col in self.CATEGORICAL_FEATURES:
                vals = df[col].unique()
                for val in vals:
                    if pd.isnull(val):
                        continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1  
            else:
                self.feature_index_[col] = last_idx
                last_idx += 1
        print(self.feature_index_)
        return self

    def fit_transform(self, df, CATEGORICAL_FEATURES, y=None):
        self.fit(df, CATEGORICAL_FEATURES, y)
        return self.transform(df)

    def transform_row_(self, row, CATEGORICAL_FEATURES):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            name = '{}_{}'.format(col, val)
            if col in self.CATEGORICAL_FEATURES:
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                
            
            else:
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, CATEGORICAL_FEATURES) for idx, row in df.iterrows()})
