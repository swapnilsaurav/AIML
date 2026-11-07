"""Weeks 5–7: reproducible synthetic churn classification lab.
Run: python lab.py   (or python baseline.py / python preprocessed.py)
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import sklearn
ROOT = Path(__file__).resolve().parent
SEED = 42

def generate_data():
    """All predictors describe information available at a fixed snapshot date."""
    r = np.random.default_rng(SEED)
    n = 10000
    income = r.lognormal(10.8, .65, n)
    burden = r.uniform(.02, .8, n)
    tenure = r.integers(1, 61, n)
    orders = r.integers(5, 101, n)
    return_rate = r.beta(1.5, 8, n)
    returns = r.binomial(orders, return_rate)
    hour = r.integers(0, 24, n)
    days = r.integers(0, 100, n)
    complaint = r.binomial(1, .35, n)
    spend = r.lognormal(8, .8, n)
    std = spend * r.uniform(.05, 1, n)
    text = np.where(complaint, r.choice(['I am unhappy with the service', 'Poor service and unresolved complaint', 'I want to cancel my account'], n), r.choice(['Happy with the service', 'Everything is good', 'Thank you for your help'], n))
    last_date = pd.Timestamp('2026-10-31') - pd.to_timedelta(days, unit='D') + pd.to_timedelta(hour, unit='h')
    # A known, intentionally engineered teaching process, not real customer data.
    logit = (-4.4 + 3.8*(burden >= .35) + 3.0*burden*(tenure <= 6)
             + 5.0*returns/orders + 1.8*complaint + 1.0*(hour < 6)
             + .018*days + .8*std/spend)
    prob = 1/(1+np.exp(-logit))
    y = r.binomial(1, prob)
    d = pd.DataFrame(dict(customer_id=[f'C{i:05d}' for i in range(n)],
        age=r.integers(20, 71, n), monthly_income=income, monthly_debt=income*burden,
        tenure_months=tenure, orders_90d=orders, returns_90d=returns,
        total_spend_90d=spend, spend_std_90d=std,
        last_purchase_timestamp=last_date.astype(str), feedback_text=text,
        region=r.choice(['North','South','East','West'], n),
        plan_type=r.choice(['Basic','Plus','Premium'], n),
        favorite_color=r.choice(['Red','Green','Blue','Yellow'], n),
        browser_type=r.choice(['Chrome','Firefox','Safari','Edge'], n),
        phone_type=r.choice(['Android','iOS','Other'], n),
        lucky_number=r.integers(1, 1000, n),
        income_copy=income*1.01, constant_field=1))
    for i in range(1, 21):
        d[f'noise_{i:02d}'] = r.normal(0, 1, n)
    # Small realistic missingness; outcome is generated before corruption.
    for col in ['monthly_income','monthly_debt','total_spend_90d','spend_std_90d']:
        d.loc[r.random(n)<.01, col] = np.nan
    d['churn'] = y
    assert d.shape == (10000, 40)
    d.to_csv(ROOT/'customer_churn_10000.csv', index=False)
    return d

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Domain rules fixed before evaluation; no target or test statistics used."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        z = X.copy()
        t = pd.to_datetime(z['last_purchase_timestamp'])
        z['purchase_hour'] = t.dt.hour
        z['purchase_day_of_week'] = t.dt.dayofweek
        z['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
        z['is_night'] = (t.dt.hour < 6).astype(int)
        z['recency_days'] = (pd.Timestamp('2026-11-01')-t).dt.total_seconds()/86400
        z['text_length'] = z['feedback_text'].fillna('').str.len()
        z['complaint_flag'] = z['feedback_text'].fillna('').str.contains('unhappy|poor|complaint|cancel', case=False, regex=True).astype(int)
        z['sentiment_score'] = np.where(z['complaint_flag'], -1, 1)
        for name, num, den in [('debt_to_income','monthly_debt','monthly_income'),
                               ('return_ratio','returns_90d','orders_90d'),
                               ('avg_order_value','total_spend_90d','orders_90d'),
                               ('spend_variability','spend_std_90d','total_spend_90d')]:
            z[name] = z[num]/z[den].replace(0, np.nan)
        z['new_customer'] = (z['tenure_months']<=6).astype(int)
        z['risk_interaction'] = z['debt_to_income']*z['new_customer']
        # Preserve missing ratios as missing rather than falsely labelling low risk.
        z['high_debt_burden'] = np.where(z['debt_to_income'].isna(), np.nan, (z['debt_to_income']>=.35).astype(float))
        return z.replace([np.inf,-np.inf],np.nan)

# Week 7: a priori domain selection. No correlations with held-out labels.
# Omit irrelevant noise/categories, ID, duplicate income, constant and raw text/date.
SELECTED = ['tenure_months','debt_to_income','return_ratio','spend_variability',
            'recency_days','is_night','complaint_flag','risk_interaction','high_debt_burden']

def build_pipeline(X, mode):
    if mode == 'baseline':
        # Minimum compatibility conversion: median fill, scaling and OHE.
        # Absolute timestamps and whole feedback strings do NOT extract semantics.
        work = X.drop(columns=['customer_id']).copy()
        work['last_purchase_timestamp'] = pd.to_datetime(work['last_purchase_timestamp']).astype('int64')/1e9
        nums = work.select_dtypes(include='number').columns.tolist()
        cats = work.select_dtypes(exclude='number').columns.tolist()
        transform = ColumnTransformer([
            ('numeric', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]), nums),
            ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False), cats)])
        return work, Pipeline([('prepare',transform),('model',LogisticRegression(C=1, max_iter=2000, random_state=SEED))])
    cols = SELECTED if mode=='preprocessed' else None
    engineered = FeatureBuilder().transform(X)
    if cols is None: # optional ablation: keep added features plus irrelevant features
        cols = engineered.select_dtypes(include='number').columns.tolist()
    transform = ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),cols)])
    return X, Pipeline([('engineer',FeatureBuilder()),('prepare',transform),('model',LogisticRegression(C=1,max_iter=2000,random_state=SEED))])

def run(mode='all'):
    d = pd.read_csv(ROOT/'customer_churn_10000.csv') if (ROOT/'customer_churn_10000.csv').exists() else generate_data()
    X,y = d.drop(columns='churn'),d.churn
    train,test = train_test_split(np.arange(len(d)),test_size=.25,random_state=SEED,stratify=y)
    modes = ['baseline','engineered_all','preprocessed'] if mode=='all' else [mode]
    rows=[]
    for name in modes:
        # Report median of three repeated fits on the SAME split, not best trial.
        timing=[]
        for repeat in range(3):
            start=perf_counter()
            work, pipe=build_pipeline(X,name)
            prep_setup=perf_counter()-start
            start=perf_counter(); pipe.fit(work.iloc[train],y.iloc[train]); fit=perf_counter()-start
            start=perf_counter(); pred=pipe.predict(work.iloc[test]); proba=pipe.predict_proba(work.iloc[test])[:,1]; infer=perf_counter()-start
            timing.append((prep_setup,fit,infer))
        med=np.median(timing,axis=0)
        row=dict(experiment=name,accuracy=accuracy_score(y.iloc[test],pred),f1=f1_score(y.iloc[test],pred),roc_auc=roc_auc_score(y.iloc[test],proba),setup_seconds=med[0],fit_seconds=med[1],predict_and_probability_seconds=med[2],total_seconds=sum(med),model_features=pipe.named_steps['model'].n_features_in_)
        rows.append(row)
        print('\n',name, json.dumps(row,indent=2))
        print('Confusion matrix [rows=true, columns=predicted]:\n',confusion_matrix(y.iloc[test],pred))
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/f'results_{mode}.csv',index=False)
    if mode=='all':
        print('\nAccuracy gain (percentage points):',100*(out.iloc[2].accuracy-out.iloc[0].accuracy))
        print('Majority-class accuracy:',y.iloc[test].value_counts(normalize=True).max())
        (ROOT/'environment.json').write_text(json.dumps(dict(numpy=np.__version__,pandas=pd.__version__,scikit_learn=sklearn.__version__,seed=SEED,train_rows=len(train),test_rows=len(test),positive_rate=float(y.mean())),indent=2))
    return out

if __name__=='__main__':
    run()
