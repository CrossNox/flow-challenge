from sklearn.model_selection import train_test_split

def sample_users(df, prop=0.2):
    train_users, test_users = train_test_split(df.account_id.unique(), test_size=prop)
    return df[df.account_id.isin(set(train_users))], df[df.account_id.isin(set(test_users))]