
# crop_risk_backend1.py
import pandas as pd
import numpy as np
from itertools import combinations

column_mapping = {
    "Sum_Insured": ["Sum_insured", "SI", "suminsured_ha", "insured_amount","sum_insured_per_ha"],
    "Area_Ha": ["area", "area under crop", "crop_area", "plot_size"],
    "Premium": ["premium", "premium_amount", "insurance_premium"],    
    "year":["year","years","YEARS","YEAR","Year","Years"]
}

group_cols = ['State', 'Cluster', 'District', 'Tehsil', 'IU_id', 'Crop']
def rename_columns_with_mapping(df, column_mapping):
    """
    Rename columns based on a mapping dictionary
    
    Parameters:
    df: pandas DataFrame
    column_mapping: dict where keys are target names and values are lists of possible patterns
    """
    new_columns = {}
    
    for col in df.columns:
        col_lower = col.lower()
        for target_name, patterns in column_mapping.items():
            # Check if column matches any of the patterns
            #if any(pattern.lower() in col.lower() for pattern in patterns):
            if any(pattern.lower() == col_lower for pattern in patterns):
                new_columns[col] = target_name
                break  # Stop checking once a match is found
    
    return df.rename(columns=new_columns)

def simulate_cluster_risks(yield_df, thresh_df, n_sims=10000, indemnity=0.1):
    """
    Simulate yield-based payouts for multiple districts/clusters.
    """

    # Merge yield and threshold files on keys
    merge_cols = ["State","Cluster", "District", "Crop"]
    df = pd.merge(yield_df, thresh_df, on=merge_cols, how="left")
    #print("df columns1 ",df.columns)
    df = rename_columns_with_mapping(df, column_mapping)
    print("df columns2 ",df.columns)
    # Handle missing values and area
    if "Area_Ha" not in df.columns:
        df["Area_Ha"] = 1.0  # assume equal area
    if "Sum_Insured" not in df.columns:
        raise ValueError("Threshold file must include 'Sum Insured' column.")
    if "cap_default_pct" not in df.columns:
        df["cap_default_pct"]=0
    df["cap_default_pct"].fillna(100, inplace=True)
    df["cap_default_pct"]=df["cap_default_pct"]/100
    
    cols = [c.lower() for c in df.columns]
    
    indemnity_factor = 1.0 - indemnity
    payr=np.maximum(0,(df.Threshold*indemnity_factor-df.Yield_kg_ha)/df.Threshold)
    payr=np.clip(payr,0,1)
    df["Payout"]=np.clip(df["Sum_Insured"]*payr,0,df["Sum_Insured"]*df["cap_default_pct"])
    df["Payout_area"]=df["Payout"]*df["Area_Ha"]
    df["SI_area"]=df["Sum_Insured"]*df["Area_Ha"]
    # Identify year columns
    #year_cols = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip())==4]
    # Aggregate data to get 1 row per IU with its key metrics
    # We use 'first' for Threshold/Sum Insured/Area assuming they are constant per IU


    iu_summary = df.groupby(group_cols).agg({
        'Yield_kg_ha': ['mean', 'std'],
        'Threshold': 'first',
        'Sum_Insured': 'first',
        'Area_Ha': 'mean',
        'Payout':['mean','std'],
        'cap_default_pct':'mean'
    }).reset_index()
    
    # Flatten multi-index columns
    iu_summary.columns = group_cols + ['mean_yield', 'std_yield', 'threshold', 'sum_insured', 'area',
                                       'mean_payout','std_payout','cap_default_pct']
    
    # Handle zero variance (your 10% rule)
    iu_summary['std_yield'] = np.where(
        iu_summary['std_yield'] == 0, 
        iu_summary['mean_yield'] * 0.1, 
        iu_summary['std_yield']
    )  
    group_colsd = [col for col in group_cols if col not in ['IU_id','Tehsil']]
    group_colsd.append('year')
    district_summary=df.groupby(group_colsd).agg( {
        'Yield_kg_ha':'mean',        
        'SI_area':'sum',
        'Area_Ha': 'sum',
        'Payout_area':'sum',        
        }).reset_index()
    district_summary["Burn_rate"]=district_summary["Payout_area"]/district_summary["SI_area"]
    
    group_colscl = [col for col in group_colsd if col not in ['District']]
    cluster_summary=df.groupby(group_colscl).agg( {
        'Yield_kg_ha':'mean',        
        'SI_area':'sum',
        'Area_Ha': 'sum',
        'Payout_area':'sum',        
        }).reset_index()
    cluster_summary["Burn_rate"]=cluster_summary["Payout_area"]/cluster_summary["SI_area"]
    num_ius = len(iu_summary)

    # Generate a Normal Distribution (Mean=0, STD=1) of shape (num_ius, n_sims)
    random_shocks = np.random.standard_normal(size=(num_ius, n_sims))
    
    # Scale shocks to the specific mean and std of each IU
    # Yields cannot be negative, so we clip at 0
    sim_matrix = (iu_summary['mean_yield'].values[:, None] + 
                  random_shocks * iu_summary['std_yield'].values[:, None])
    sim_matrix = np.maximum(0, sim_matrix)
    #print("mean yields",mean_yields)

    
    # 1. Calculate Payout Ratio
    # Using broadcasting: (IUs, 1) vs (IUs, 1000)
    thresh = iu_summary['threshold'].values[:, None]
    payout_ratio = np.maximum(0, (thresh * indemnity_factor - sim_matrix) / thresh)
    payout_ratio = np.clip(payout_ratio, 0, 1)
    
    # 2. Calculate Payouts (Total Rupees/Currency)
    si = iu_summary['sum_insured'].values[:, None]
    area = iu_summary['area'].values[:, None]
    payouts = payout_ratio * si * area
    pct=iu_summary['cap_default_pct'].values[:,None]
    payouts=np.clip(payouts,0,pct*si*area)
    
    sim_cols = [f's_{i}' for i in range(n_sims)]
    payouts_df=pd.DataFrame(payouts, columns=sim_cols)
    iu_summary = pd.concat([iu_summary, payouts_df], axis=1)
    # 3. Calculate Burn Rate per IU
    # Burn Rate = Average Payout / Total Sum Insured
    
    avg_payout_per_iu = np.mean(payouts, axis=1)
    iu_summary['expected_payout'] = avg_payout_per_iu
    iu_summary['totalsi']=iu_summary['sum_insured'] * iu_summary['area']
    iu_summary['burn_rate'] = avg_payout_per_iu / iu_summary['totalsi']
    
    group_colsd = [col for col in group_cols if col not in ['Crop']]
    agg_dict = {
    'totalsi': 'sum',
    #'expected_payout': 'sum'
    }
    for col in sim_cols:
        agg_dict[col] = 'sum'
    
    def cal_sim_summary(sim_df,group_cols,agg_dict,sim_cols):
        temp1=sim_df.groupby(group_colsd).agg(agg_dict).reset_index()
        temp=temp1[sim_cols].div(temp1['totalsi'], axis=0)
        summary_sim=temp1[group_colsd].copy()
        summary_sim['avg_payout']=temp1[sim_cols].mean(axis=1)
        summary_sim['std_payout']=temp1[sim_cols].std(axis=1)
        summary_sim['cv_payout']=summary_sim['std_payout'].div(summary_sim['avg_payout'])
        summary_sim['totalsi']=temp1['totalsi']
        summary_sim['burn_rate']=temp.mean(axis=1)
        summary_sim['burn_rate_sd']=temp.std(axis=1)
        summary_sim['cv'] = summary_sim['burn_rate_sd'] / summary_sim['burn_rate']
        percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95]
        percentile_df = temp.quantile(percentiles, axis=1).T  # Transpose to align rows with IU
        percentile_df.columns = [f'p{int(p*100)}' for p in percentiles]
        summary_sim = pd.concat([summary_sim, percentile_df], axis=1)
        summary_sim['volatility_rank']=summary_sim["cv_payout"].rank(ascending=False)
        summary_sim['profitability_rank']=summary_sim["burn_rate"].rank(ascending=True)
        summary_sim['stability_rank']=summary_sim["cv_payout"].rank(ascending=True)
        return summary_sim,temp1
    iu_summary_sim,temp1=cal_sim_summary(iu_summary, group_colsd, agg_dict,sim_cols)
    group_colsd = [col for col in group_cols if col not in ['Tehsil','IU_id','Crop']]
    district_summary_sim,temp1=cal_sim_summary(temp1, group_colsd, agg_dict,sim_cols)
    group_colsd = [col for col in group_cols if col not in ['District','Tehsil','IU_id','Crop']]
    cluster_summary_sim,temp1=cal_sim_summary(temp1, group_colsd, agg_dict,sim_cols)
      
    
    return district_summary_sim, cluster_summary_sim


def find_top_combinations(cluster_summary, top_n=5):
    """Find top combinations of clusters."""
    combos = []
    cluster_names = cluster_summary["Cluster"].unique()

    for combo in combinations(cluster_names, 2):
        df2 = cluster_summary[cluster_summary["Cluster"].isin(combo)]
        combo_burn = df2["burn_rate"].mean()
        combo_cv = df2["cv_payout"].mean()
        combos.append({
            "Combo": f"{combo[0]} + {combo[1]}",
            "mean_Burn": combo_burn,
            "cv": combo_cv
        })

    return pd.DataFrame(combos).sort_values("cv").head(top_n)

yield_file=r"D:/testapp/python/crop_insurance_portfolio/yield.csv"
thresh_file=r"D:/testapp/python/crop_insurance_portfolio/Thresholds1.csv"
yield_df = pd.read_csv(yield_file)
thresh_df = pd.read_csv(thresh_file)
sims=1000
indemnity=10/100
district_summary, cluster_summary = simulate_cluster_risks(
    yield_df, thresh_df, n_sims=sims, indemnity=indemnity
)