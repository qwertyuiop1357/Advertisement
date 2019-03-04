def build_date_buf(date_pivot, left, right):
    """
    :param date_pivot:
    :param move:
    :return:
    """

    date_buf = []
    buf = range(left, right)
    for i in buf:
        date = date_pivot  + pd.Timedelta(i ,  unit = 'd')
        date_buf.append(date.strftime('%Y-%m-%d'))

    return date_buf

def _generate_historical_convrate(df):
    """
    historical conversion rate
    :param df:
    :return:
    """
    unique_date = df['date'].unique()
    col_list = [(['item_id'], 'item_convrate'), (['user_age_level', 'item_id'], 'age_item_convraterate')]

    data_buf = []
    for day in unique_date:
        date_pivot = pd.to_datetime(day)
        lag_days = _build_date_buf(date_pivot, -3, 0)

        target_df = df[df['date'].isin([day])]
        lag_df = df[df['date'].isin(lag_days)]

        if lag_df.shape[0] == 0 or target_df.shape[0] == 0:
            continue

        for cols, col_name in col_list:
            lag_g = lag_df.groupby(cols).is_trade.mean().reset_index()

            lag_cols = []
            lag_cols.extend(cols)
            lag_cols.append(col_name)

            lag_g.columns = lag_cols
            target_df = pd.merge(target_df, lag_g, on=cols, how='left').fillna(0)
        data_buf.append(target_df)

    hc_df = pd.concat(data_buf, axis=0).reset_index(drop=True)
    return hc_df


def _make_instant_feature(df):
    first, prev = -1, -1
    first_buf, prev_buf, fif_min_buf = [], [], []

    for row in df.itertuples():
        cur = row.context_timestamp

        if first == -1:
            first = row.context_timestamp

        first_buf.append(cur - first)

        if prev == -1:
            prev_buf.append(0)
            fif_min_buf.append(1)
        else:
            prev_buf.append(cur - prev)
            if cur - prev <= 15 * 60:
                fif_min_buf.append(fif_min_buf[-1] + 1)
            else:
                fif_min_buf.append(1)
        prev = cur

    df['first_to_now'] = first_buf
    df['prev_to_now'] = prev_buf
    df['recent_15_minutes'] = fif_min_buf

    return df[['instance_id', 'first_to_now', 'prev_to_now', 'recent_15_minutes']].reset_index(drop=True)


def _generate_instant_feature(df):
    """
    instant clicks
    :param df:
    :return:
    """
    sorted_df = df.sort_values('context_timestamp')

    uig = sorted_df.groupby(['user_id', 'item_id'])
    ins_g = uig[['instance_id', 'context_timestamp']].apply(_make_instant_feature)

    ins_df = pd.merge(df, ins_g.reset_index(drop=True), on='instance_id')
    return ins_df



