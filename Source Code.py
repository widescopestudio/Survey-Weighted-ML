import warnings; warnings.filterwarnings("ignore")
import os, random, time, sys
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score, brier_score_loss,
                              roc_curve, confusion_matrix)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import shap
from deap import base, creator, tools, algorithms
from fairlearn.metrics import (demographic_parity_difference,
                                equalized_odds_difference)

_ROOT = ""

ROUNDS_CONFIG = {
    2011: {
        "kr": f"{_ROOT}/BD_2011_DHS_05232026_1412_236689/BDKR61SV/BDKR61FL.SAV",
        "ir": f"{_ROOT}/BD_2011_DHS_05232026_1412_236689/BDIR61SV/BDIR61FL.SAV",
        "hr": f"{_ROOT}/BD_2011_DHS_05232026_1412_236689/BDHR61SV/BDHR61FL.SAV",
    },
    2014: {
        "kr": f"{_ROOT}/BD_2014_DHS_05232026_1412_236689/BDKR72SV/BDKR72FL.SAV",
        "ir": f"{_ROOT}/BD_2014_DHS_05232026_1412_236689/BDIR72SV/BDIR72FL.SAV",
        "hr": f"{_ROOT}/BD_2014_DHS_05232026_1412_236689/BDHR72SV/BDHR72FL.SAV",
    },
    2017: {
        "kr": f"{_ROOT}/BD_2017-18_DHS_05232026_1411_236689/BDKR7RSV/BDKR7RFL.SAV",
        "ir": f"{_ROOT}/BD_2017-18_DHS_05232026_1411_236689/BDIR7RSV/BDIR7RFL.SAV",
        "hr": f"{_ROOT}/BD_2017-18_DHS_05232026_1411_236689/BDHR7RSV/BDHR7RFL.SAV",
    },
    2022: {
        "kr": f"{_ROOT}/BD_2022_DHS_05232026_149_236689/BDKR81SV/BDKR81FL.SAV",
        "ir": f"{_ROOT}/BD_2022_DHS_05232026_149_236689/BDIR81SV/BDIR81FL.SAV",
        "hr": f"{_ROOT}/BD_2022_DHS_05232026_149_236689/BDHR81SV/BDHR81FL.SAV",
    },
}

OUTPUT_DIR = "/kaggle/working/"

RUN_MICE             = True
N_MICE_IMPUTATIONS   = 5
N_BOOT               = 500
RUN_ABLATION         = True
RUN_PANEL_ANALYSIS   = True
RUN_SPATIAL_SENS     = True
SEED                 = 42

GA_POP  = 50
GA_NGEN = 45

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
t0 = time.time()

def elapsed(): return f"[{time.time()-t0:6.1f}s]"
def hdr(s):    print(f"\n{'='*72}\n  {s}\n{'='*72}")
def sub(s):    print(f"\n  ── {s}")

BLUE   = "#1f4e79"; TEAL   = "#006d77"; CORAL  = "#c0392b"
AMBER  = "#e67e22"; GREEN  = "#27ae60"; PURPLE = "#6c3483"
GREY10 = "#1a1a1a"; GREY40 = "#666666"; GREY70 = "#b3b3b3"
PAL6   = [BLUE, TEAL, CORAL, AMBER, GREEN, PURPLE]
ROUND_COLORS = {2011: PURPLE, 2014: AMBER, 2017: TEAL, 2022: BLUE}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {name}")

KR_COLS_CORE = [
    "V001","V002","V003","V005","V021","V022","V023","V024","V025","V040",
    "CASEID","BIDX","HWIDX",
    "B4","B5","B8","B11","B19",
    "HW1","HW2","HW3","HW13","HW70","HW71","HW72","HW73",
    "V437","V438","V445",
    "V012","V106","V107","V133","V149",
    "V113","V116","V119","V121","V136","V137","V151","V152","V160",
    "V190","V191",
    "M4","M14","M15","M17","M18","M19","M45","M46",
    "H11","H22","H31","H34","H43",
    "V414A","V414E","V414F","V414G","V414H",
    "V414I","V414J","V414K","V414N","V414O",
    "V743A","V743B","V743D","V743F",
    "V744A","V744B","V744C","V744D","V744E",
]
KR_COLS_NEW = ["SDIST","V169A","V169B","V170","V177"]

IR_COLS = [
    "V001","V002","V003",
    "V743A","V743B","V743D","V743F",
    "V744A","V744B","V744C","V744D","V744E",
    "V701","V702","V717","V501","V502","V531",
    "V312","V313",
    "V169A","V169B","V170",
]

HR_COLS = [
    "HV001","HV002",
    "HV225","HV230A","HV232","HV235","HV237","HV241","HV244",
    "HV247","HV270","HV213","HV214","HV215","HV216",
]

PARAM_SPACE = {
    "n_estimators":     [100, 200, 300, 400, 500],
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.005, 0.01, 0.05, 0.1, 0.15],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma":            [0, 0.05, 0.1, 0.3, 0.5],
    "reg_alpha":        [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
    "scale_pos_weight": [1.0, 2.0, 3.0, 3.5, 4.0],
}
PNAMES = list(PARAM_SPACE.keys())
PVALS  = list(PARAM_SPACE.values())

DEFAULT_PARAMS = {
    "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
    "scale_pos_weight": 3.5,
}

DIV_LABELS = {
    1:"Barishal", 2:"Chattogram", 3:"Dhaka", 4:"Khulna",
    5:"Mymensingh", 6:"Rajshahi", 7:"Rangpur", 8:"Sylhet",
}

DOMAIN_COLORS = {
    "Maternal anthropometry": "#1D9E75",
    "Autonomy/Husband":       "#9B59B6",
    "ANC/Delivery":           "#E67E22",
    "Child morbidity":        "#E74C3C",
    "Wealth":                 "#378ADD",
    "WASH":                   "#16A085",
    "Child":                  TEAL,
    "Geography/HH":           "#95A5A6",
}

def feat_domain(f):
    if f in ["mat_height_cm","mat_bmi","mat_short","mat_underweight",
             "mat_overweight","mat_age","mat_teen","mat_height_sq"]:
        return "Maternal anthropometry"
    if any(x in f for x in ["autonomy","violence","husb_"]): return "Autonomy/Husband"
    if any(x in f for x in ["anc","facility","c_section","iron",
                              "breastfed","small_at","birth_weight"]):
        return "ANC/Delivery"
    if any(x in f for x in ["diarrhea","fever","cough","vitamin","deworm","diet"]):
        return "Child morbidity"
    if f in ["wealth_idx","poor_hh","poorest"]: return "Wealth"
    if any(x in f for x in ["WASH","water","sanit","soap","toilet"]): return "WASH"
    if any(x in f for x in ["age","birth_order","birth_interval",
                              "child_","first_born","short_interval"]):
        return "Child"
    return "Geography/HH"

def decode_ga(ind):
    return {n: PVALS[i][int(ind[i]) % len(PVALS[i])]
            for i, n in enumerate(PNAMES)}

def improved_water(x):
    if pd.isna(x): return np.nan
    return 1.0 if int(x) in {11,12,13,14,21,31,41,51,71} else 0.0

def improved_sanit(x):
    if pd.isna(x): return np.nan
    return 1.0 if int(x) in {11,12,13,14,15,21,22,41} else 0.0

def mice_impute(X, n_imp=5, seed=42):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imps = []
    for i in range(n_imp):
        imp = IterativeImputer(max_iter=10, random_state=seed+i,
                               initial_strategy="median")
        imps.append(imp.fit_transform(X))
    return imps

def rubin_pool(point_estimates, variances):
    m = len(point_estimates)
    Q = np.mean(point_estimates)
    W = np.mean(variances)
    B = np.var(point_estimates, ddof=1)
    T = W + (1 + 1/m)*B
    return Q, np.sqrt(T)

def bootstrap_metrics(y_true, y_prob, y_pred, weights, psu, strata,
                       n_boot=N_BOOT, seed=42):
    rng  = np.random.default_rng(seed)
    boot = {m: [] for m in ["auc","f1","sensitivity","specificity","ppv","brier"]}

    for _ in range(n_boot):
        idx = []
        for s in np.unique(strata):
            sm   = strata == s
            psus = np.unique(psu[sm])
            if len(psus) < 2: continue
            for p in rng.choice(psus, size=len(psus), replace=True):
                idx.extend(np.where((strata==s) & (psu==p))[0])
        if len(idx) < 10: continue
        idx = np.array(idx)
        yt, yp, yd, wt = y_true[idx], y_prob[idx], y_pred[idx], weights[idx]
        if yt.sum() < 2 or (1-yt).sum() < 2: continue
        try:
            boot["auc"].append(roc_auc_score(yt, yp, sample_weight=wt))
            boot["f1"].append(f1_score(yt, yd, sample_weight=wt, zero_division=0))
            boot["sensitivity"].append(recall_score(yt, yd, sample_weight=wt, zero_division=0))
            boot["specificity"].append(recall_score(yt, yd, sample_weight=wt, pos_label=0, zero_division=0))
            boot["ppv"].append(precision_score(yt, yd, sample_weight=wt, zero_division=0))
            boot["brier"].append(brier_score_loss(yt, yp, sample_weight=wt))
        except: pass

    def pt(m):
        if m == "auc":         return roc_auc_score(y_true, y_prob, sample_weight=weights)
        if m == "f1":          return f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        if m == "sensitivity": return recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        if m == "specificity": return recall_score(y_true, y_pred, sample_weight=weights, pos_label=0, zero_division=0)
        if m == "ppv":         return precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        if m == "brier":       return brier_score_loss(y_true, y_prob, sample_weight=weights)

    results = {}
    for m, vals in boot.items():
        vals = np.array(vals)
        p    = pt(m)
        results[m] = (round(p, 4),
                      round(np.percentile(vals, 2.5), 4),
                      round(np.percentile(vals, 97.5), 4))
    return results

def opt_thresh(yt, yp, w):
    best_t, best_f = 0.30, 0
    for t in np.linspace(0.15, 0.65, 100):
        f = f1_score(yt, (yp>=t).astype(int), sample_weight=w, zero_division=0)
        if f > best_f: best_f, best_t = f, t
    return best_t

def find_thresh_tpr(proba, y_true, weights, target_tpr):
    best_t, best_gap = 0.30, 999
    pos = y_true == 1
    if pos.sum() < 3: return 0.30
    for t in np.linspace(0.10, 0.80, 150):
        tpr = np.average((proba[pos] >= t), weights=weights[pos])
        if abs(tpr - target_tpr) < best_gap:
            best_gap = abs(tpr - target_tpr); best_t = t
    return best_t

def apply_fairness_calibration(p_train, y_train, w_train, wl_train,
                                p_test, wl_test):
    overall_tpr = np.average(p_train[y_train==1] >= 0.30,
                             weights=w_train[y_train==1])
    group_thresh = {}
    for q in range(1, 6):
        m = wl_train == q
        group_thresh[q] = find_thresh_tpr(
            p_train[m], y_train[m], w_train[m], overall_tpr
        ) if m.sum() > 10 else 0.30

    yd_fair = np.zeros(len(p_test), dtype=int)
    for q in range(1, 6):
        m = wl_test == q
        yd_fair[m] = (p_test[m] >= group_thresh[q]).astype(int)
    return yd_fair, group_thresh

def safe_shap_series(shap_values, feature_names, label=""):
    n_out  = shap_values.shape[1]
    n_feat = len(feature_names)
    if n_out != n_feat:
        print(f"    [safe_shap] {label}: SHAP width={n_out} vs index={n_feat} "
              f"— trimming index to {n_out}")
        idx = list(feature_names)[:n_out]
    else:
        idx = list(feature_names)
    return pd.Series(np.abs(shap_values).mean(axis=0), index=idx)

def net_benefit(y_true, y_prob, threshold):
    n    = len(y_true)
    ypred = (y_prob >= threshold).astype(int)
    tp   = ((ypred==1) & (y_true==1)).sum()
    fp   = ((ypred==1) & (y_true==0)).sum()
    return (tp/n) - (fp/n) * (threshold / (1-threshold))

def bootstrap_auc_diff_pval(y, p1, p2, w, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    obs = (roc_auc_score(y, p1, sample_weight=w) -
           roc_auc_score(y, p2, sample_weight=w))
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        try:
            d = (roc_auc_score(y[idx], p1[idx], sample_weight=w[idx]) -
                 roc_auc_score(y[idx], p2[idx], sample_weight=w[idx]))
            diffs.append(d)
        except: pass
    diffs = np.array(diffs)
    mean_d = diffs.mean()
    pval = np.mean(np.abs(diffs - mean_d) >= np.abs(obs))
    return obs, max(pval, 1/n_boot)

def load_and_engineer(kr_path, ir_path, hr_path, survey_year):
    sub(f"Loading BDHS {survey_year}")

    def safe_read(path):
        df, _ = pyreadstat.read_sav(path, apply_value_formats=False)
        df.columns = [c.upper() for c in df.columns]
        return df

    df_kr = safe_read(kr_path)
    df_ir = safe_read(ir_path)
    df_hr = safe_read(hr_path)

    df = df_kr.copy()
    df = df[df["B5"] == 1]
    df = df[df["HW1"].between(0, 59)]
    df = df[df["HW70"].notna() & (df["HW70"] < 9990)]
    if "HW13" in df.columns:
        df = df[df["HW13"] == 0]
    print(f"    {survey_year} after eligibility: n={len(df):,}")

    ir_keep = [c for c in IR_COLS if c in df_ir.columns]
    df = df.merge(df_ir[ir_keep], on=["V001","V002","V003"], how="left")

    hr_df = df_hr.copy()
    if "HV001" in hr_df.columns: hr_df = hr_df.rename(columns={"HV001":"V001","HV002":"V002"})
    hr_keep = [c for c in hr_df.columns if c.startswith("HV") or c in ["V001","V002"]]
    df = df.merge(hr_df[hr_keep].drop_duplicates(["V001","V002"]),
                  on=["V001","V002"], how="left", suffixes=("","_hr"))

    df["survey_weight"] = df["V005"] / 1_000_000
    df["psu"]           = df["V021"].astype(float)
    df["strata"]        = df["V023"].astype(float)
    df["survey_year"]   = survey_year

    df["stunted"]        = (df["HW70"] < -200).astype(int)
    df["severe_stunted"] = (df["HW70"] < -300).astype(int)
    df["underweight"]    = np.where(df["HW71"].notna() & (df["HW71"]<9990),
                                    (df["HW71"]<-200).astype(int), np.nan)
    df["wasted"]         = np.where(df["HW72"].notna() & (df["HW72"]<9990),
                                    (df["HW72"]<-200).astype(int), np.nan)
    df["any_malnut"]     = ((df["stunted"]==1) |
                            (df["underweight"].fillna(0)==1) |
                            (df["wasted"].fillna(0)==1)).astype(int)

    df["child_age_mo"]   = df["HW1"].astype(float)
    df["age_group"]      = pd.cut(df["child_age_mo"],
                                   bins=[-1,5,11,23,35,47,59],
                                   labels=[0,1,2,3,4,5]).astype(float)
    df["child_female"]   = (df["B4"] == 2).astype(float)
    df["birth_order"]    = df["BIDX"].clip(1, 6).astype(float)
    df["first_born"]     = (df["BIDX"] == 1).astype(float)
    df["birth_interval"] = df["B11"].astype(float)
    df["short_interval"] = (df["B11"] < 24).astype(float)

    df["mat_height_cm"]   = np.where(df["V438"] < 9990, df["V438"]/10, np.nan)
    df["mat_bmi"]         = np.where(df["V445"] < 9990, df["V445"]/100, np.nan)
    df["mat_short"]       = (df["mat_height_cm"] < 145).astype(float)
    df["mat_underweight"] = (df["mat_bmi"] < 18.5).astype(float)
    df["mat_overweight"]  = (df["mat_bmi"] >= 25).astype(float)
    df["mat_age"]         = df["V012"].astype(float)
    df["mat_teen"]        = (df["V012"] < 20).astype(float)
    df["mat_height_sq"]   = df["mat_height_cm"] ** 2

    df["mat_edu_level"]      = df["V106"].clip(0, 3)
    df["mat_edu_yrs"]        = df["V133"].clip(0, 20)
    df["mat_no_edu"]         = (df["V106"] == 0).astype(float)
    df["mat_secondary_plus"] = (df["V106"] >= 2).astype(float)

    for col, alias in [("V743A","autonomy_health"),("V743B","autonomy_purchases"),
                       ("V743D","autonomy_visits"),("V743F","autonomy_money")]:
        if col in df.columns:
            df[alias] = df[col].apply(
                lambda x: 1.0 if x in [1,2] else (0.0 if x in [3,4,5,6] else np.nan))
    auto_cols = [c for c in ["autonomy_health","autonomy_purchases",
                              "autonomy_visits","autonomy_money"] if c in df.columns]
    df["autonomy_score"] = df[auto_cols].mean(axis=1) if auto_cols else np.nan

    v_cols = [c for c in ["V744A","V744B","V744C","V744D","V744E"] if c in df.columns]
    if v_cols:
        for c in v_cols:
            df[c] = df[c].apply(lambda x: 1.0 if x==1 else (0.0 if x==0 else np.nan))
        df["violence_acceptance"] = df[v_cols].sum(axis=1)
        df["accepts_violence"]    = (df["violence_acceptance"] > 0).astype(float)

    if "V701" in df.columns:
        df["husb_edu_level"]      = df["V701"].clip(0, 3)
        df["husb_no_edu"]         = (df["V701"] == 0).astype(float)
        df["husb_secondary_plus"] = (df["V701"] >= 2).astype(float)

    df["anc_visits"]        = df["M14"].apply(
        lambda x: np.nan if pd.isna(x) or x>=97 else float(x))
    df["anc_4plus"]         = (df["anc_visits"] >= 4).astype(float)
    df["anc_none"]          = (df["anc_visits"] == 0).astype(float)
    df["facility_delivery"] = df["M15"].apply(
        lambda x: 1.0 if not pd.isna(x) and 10<=int(x)<=19
                  else (0.0 if not pd.isna(x) and int(x)>=20 else np.nan))
    df["c_section"]         = (df["M17"] == 1).astype(float)
    df["small_at_birth"]    = df["M18"].apply(
        lambda x: 1.0 if not pd.isna(x) and int(x)>=4
                  else (0.0 if not pd.isna(x) else np.nan))
    df["iron_tablets"]      = (df["M45"] == 1).astype(float)
    df["ever_breastfed"]    = df["M4"].apply(
        lambda x: 0.0 if not pd.isna(x) and x==0
                  else (1.0 if not pd.isna(x) else np.nan))

    df["had_diarrhea"]    = df["H11"].apply(
        lambda x: 1.0 if not pd.isna(x) and x==2
                  else (0.0 if not pd.isna(x) and x==0 else np.nan))
    df["had_fever"]       = (df["H22"] == 1).astype(float)
    df["had_cough"]       = df["H31"].apply(
        lambda x: 1.0 if not pd.isna(x) and x==2
                  else (0.0 if not pd.isna(x) and x==0 else np.nan))
    df["morbidity_score"] = (df["had_diarrhea"].fillna(0) +
                              df["had_fever"].fillna(0) +
                              df["had_cough"].fillna(0))
    df["vitamin_a"]       = (df["H34"] == 1).astype(float)
    df["dewormed"]        = (df["H43"] == 1).astype(float)

    diet_cols = [c for c in ["V414A","V414E","V414F","V414G","V414H",
                               "V414I","V414J","V414K","V414N","V414O"]
                 if c in df.columns]
    if diet_cols:
        for c in diet_cols: df[c+"_b"] = (df[c]==1).astype(float)
        df["dietary_diversity"]  = np.where(
            df["child_age_mo"].between(6,23),
            df[[c+"_b" for c in diet_cols]].sum(axis=1), np.nan)
        df["low_diet_diversity"] = (df["dietary_diversity"] < 4).astype(float)

    df["improved_water"] = df["V113"].apply(improved_water)
    df["improved_sanit"] = df["V116"].apply(improved_sanit)
    hv232 = df["HV232"] if "HV232" in df.columns else pd.Series(np.nan, index=df.index)
    hv237 = df["HV237"] if "HV237" in df.columns else pd.Series(np.nan, index=df.index)
    hv225 = df["HV225"] if "HV225" in df.columns else pd.Series(np.nan, index=df.index)
    df["soap_present"]   = (hv232 == 1).astype(float)
    df["water_treated"]  = (hv237 == 1).astype(float)
    df["toilet_private"] = (hv225 == 0).astype(float)
    df["WASH_score"]     = (df["improved_water"].fillna(0) +
                             df["improved_sanit"].fillna(0) +
                             df["soap_present"].fillna(0)   +
                             df["water_treated"].fillna(0)  +
                             df["toilet_private"].fillna(0))

    df["wealth_idx"]      = df["V190"].clip(1, 5)
    df["poor_hh"]         = (df["V190"] <= 2).astype(float)
    df["poorest"]         = (df["V190"] == 1).astype(float)
    df["has_electricity"] = (df["V119"] == 1).astype(float)
    df["has_tv"]          = (df["V121"] == 1).astype(float)
    df["has_mobile"]      = (df["V169A"] == 1).astype(float) \
                             if "V169A" in df.columns else np.nan
    df["has_bank"]        = (df["V170"] == 1).astype(float) \
                             if "V170" in df.columns else np.nan
    df["hh_size"]         = df["V136"].astype(float)
    hv216 = df["HV216"] if "HV216" in df.columns else pd.Series(np.nan, index=df.index)
    df["crowding_ratio"]  = (df["V136"] / hv216.replace(0, np.nan)).clip(1, 10)
    df["urban"]           = (df["V025"] == 1).astype(float)
    df["division"]        = df["V024"].astype(float)
    df["district"]        = df["SDIST"].astype(float) \
                             if "SDIST" in df.columns else np.nan
    df["cluster_id"]      = df["V001"].astype(float)
    df["altitude_m"]      = df["V040"].astype(float)
    df["hoh_female"]      = (df["V151"] == 2).astype(float)

    FEATURES_ALL = [
        "child_age_mo","age_group","child_female","birth_order","first_born",
        "birth_interval","short_interval",
        "mat_height_cm","mat_bmi","mat_short","mat_underweight","mat_overweight",
        "mat_age","mat_teen","mat_height_sq",
        "mat_edu_level","mat_edu_yrs","mat_no_edu","mat_secondary_plus",
        "autonomy_score","autonomy_health","autonomy_purchases",
        "accepts_violence","violence_acceptance",
        "husb_edu_level","husb_no_edu","husb_secondary_plus",
        "anc_visits","anc_4plus","anc_none","facility_delivery",
        "c_section","small_at_birth","iron_tablets","ever_breastfed",
        "had_diarrhea","had_fever","had_cough","morbidity_score",
        "vitamin_a","dewormed",
        "dietary_diversity","low_diet_diversity",
        "WASH_score","improved_water","improved_sanit","soap_present","toilet_private",
        "wealth_idx","poor_hh","poorest",
        "has_electricity","has_tv","has_mobile","has_bank",
        "hh_size","crowding_ratio",
        "urban","division","altitude_m","hoh_female",
    ]
    FEATURES_CANDIDATE = [f for f in FEATURES_ALL if f in df.columns]

    META = ["survey_weight","psu","strata","survey_year",
            "V001","V002","V024","V025","V190",
            "district","cluster_id","division",
            "HW70","HW71","HW72","mat_height_cm","mat_bmi"]
    META = [c for c in META if c in df.columns]
    OUTCOMES = ["stunted","underweight","wasted","any_malnut","severe_stunted"]

    ALL_COLS = list(dict.fromkeys(FEATURES_CANDIDATE + OUTCOMES + META))
    ALL_COLS = [c for c in ALL_COLS if c in df.columns]
    df_out   = df[ALL_COLS].copy()
    df_out   = df_out.dropna(subset=["stunted","survey_weight","child_age_mo"])

    FEATURES = [
        f for f in FEATURES_CANDIDATE
        if f in df_out.columns and df_out[f].notna().sum() > 0
    ]

    w   = df_out["survey_weight"]
    prev = np.average(df_out["stunted"], weights=w) * 100
    print(f"    n={len(df_out):,}  stunting={prev:.1f}%  "
          f"features={len(FEATURES)}  "
          f"mat_ht_available={df_out['mat_height_cm'].notna().sum():,}")

    miss_pct = df_out[FEATURES].isna().mean() * 100
    high_miss = miss_pct[miss_pct > 5].sort_values(ascending=False)
    if len(high_miss):
        print(f"    Features >5% missing in {survey_year}:")
        for feat, pct in high_miss.items():
            print(f"      {feat:<30} {pct:.1f}%")

    return df_out, FEATURES


def build_harmonization_report(round_dfs, round_feats, all_features):
    rows = []
    for yr, feats in round_feats.items():
        missing_feats = [f for f in all_features if f not in feats]
        df_yr = round_dfs[yr]
        high_miss = []
        for f in feats:
            if f in df_yr.columns:
                pct = df_yr[f].isna().mean() * 100
                if pct > 10:
                    high_miss.append((f, pct))
        rows.append({
            "Survey Year": yr,
            "Total features": len(feats),
            "Missing features (set to NaN)": ", ".join(missing_feats) if missing_feats else "None",
            "High-missingness features (>10%)": "; ".join([f"{f}:{p:.0f}%" for f,p in high_miss]) if high_miss else "None",
            "n_missing_feats": len(missing_feats),
        })
    return pd.DataFrame(rows)


hdr("PART 1: Load BDHS 2011 / 2014 / 2017 / 2022")

round_dfs   = {}
round_feats = {}

for yr, paths in ROUNDS_CONFIG.items():
    df_yr, feats_yr = load_and_engineer(
        paths["kr"], paths["ir"], paths["hr"], yr)
    round_dfs[yr]   = df_yr
    round_feats[yr] = feats_yr

FEATURES = round_feats[2022]

harm_report = build_harmonization_report(round_dfs, round_feats, FEATURES)
harm_report.to_csv(os.path.join(OUTPUT_DIR, "feature_harmonization.csv"), index=False)
print("\n  Feature Harmonization Report:")
print(harm_report.to_string(index=False))

df_panel = pd.concat(round_dfs.values(), ignore_index=True, sort=False)
print(f"\n  Panel: n={len(df_panel):,} children across 4 rounds")

print("\n  Stunting prevalence by round:")
for yr in sorted(round_dfs):
    df_yr = round_dfs[yr]
    prev  = np.average(df_yr["stunted"], weights=df_yr["survey_weight"]) * 100
    print(f"    {yr}: {prev:.1f}%  (n={len(df_yr):,})")


hdr("PART 2: 2022 Development Cohort — MICE Imputation & Train/Test Split")

df22    = round_dfs[2022]

FEATS22 = [f for f in round_feats[2022] if f in df22.columns]
FEATURES = FEATS22
print(f"  FEATS22 reconciled: {len(round_feats[2022])} candidates → "
      f"{len(FEATS22)} confirmed present in df22")
dropped = [f for f in round_feats[2022] if f not in df22.columns]
if dropped:
    print(f"  Dropped (not in df22): {dropped}")

w22    = df22["survey_weight"].values
psu22  = df22["psu"].values
str22  = df22["strata"].values
wl22   = df22["V190"].fillna(3).values.astype(int)
div22  = df22["division"].fillna(3).values.astype(int)

X_raw22 = df22[FEATS22].values
y22     = df22["stunted"].values.astype(int)

miss_pct22 = pd.Series(df22[FEATS22].isna().mean()*100, index=FEATS22).sort_values(ascending=False)
miss_table = miss_pct22[miss_pct22 > 0].reset_index()
miss_table.columns = ["Feature", "Missing (%)"]
miss_table.to_csv(os.path.join(OUTPUT_DIR, "missing_data_2022.csv"), index=False)
print(f"  Features with any missingness: {(miss_pct22>0).sum()}")
print(f"  Features >10% missing: {(miss_pct22>10).sum()}")
print(f"  Features >5% missing:")
for f, p in miss_pct22[miss_pct22>5].items():
    print(f"    {f:<35} {p:.1f}%")

if RUN_MICE:
    print(f"\n  Running MICE ({N_MICE_IMPUTATIONS} imputations) ...")
    imputed_datasets = mice_impute(X_raw22, n_imp=N_MICE_IMPUTATIONS, seed=SEED)
    X22 = np.mean(imputed_datasets, axis=0)
    print(f"  MICE complete. Shape: {X22.shape}")
else:
    imputer_med = SimpleImputer(strategy="median")
    X22 = imputer_med.fit_transform(X_raw22)
    imputed_datasets = [X22]

(X_tr22, X_te22,
 y_tr22, y_te22,
 w_tr22, w_te22,
 wl_tr22,wl_te22,
 psu_tr22, psu_te22,
 str_tr22, str_te22,
 div_tr22, div_te22) = train_test_split(
    X22, y22, w22, wl22, psu22, str22, div22,
    test_size=0.20, random_state=SEED, stratify=y22)

print(f"  Train: n={len(X_tr22):,}  Test: n={len(X_te22):,}")

if RUN_MICE and len(imputed_datasets) > 1:
    sub("MICE Rubin's Rules — Per-imputation AUC")
    mice_aucs = []
    for Xi in imputed_datasets:
        Xtr_i, Xte_i, ytr_i, yte_i, wtr_i, wte_i = train_test_split(
            Xi, y22, w22, test_size=0.2, random_state=SEED, stratify=y22)
        m_tmp = xgb.XGBClassifier(n_estimators=300, max_depth=5,
                                   learning_rate=0.05, random_state=SEED,
                                   eval_metric="logloss", n_jobs=-1)
        m_tmp.fit(Xtr_i, ytr_i, sample_weight=wtr_i)
        p_tmp = m_tmp.predict_proba(Xte_i)[:,1]
        mice_aucs.append(roc_auc_score(yte_i, p_tmp, sample_weight=wte_i))
    pooled_auc, pooled_se = rubin_pool(mice_aucs, [0.0001]*N_MICE_IMPUTATIONS)
    print(f"\n  Per-imputation AUCs: {[round(a,4) for a in mice_aucs]}")
    print(f"  AUC range: {min(mice_aucs):.4f}–{max(mice_aucs):.4f}")
    print(f"  Rubin-pooled AUC: {pooled_auc:.4f} ± {pooled_se:.4f}")


hdr("PART 3: Genetic Algorithm Optimisation (GA)")

cv_ga = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate_ga(individual):
    params = decode_ga(individual)
    model  = xgb.XGBClassifier(**params, eval_metric="logloss",
                                random_state=SEED, n_jobs=-1)
    aucs = []
    for ti, vi in cv_ga.split(X_tr22, y_tr22):
        model.fit(X_tr22[ti], y_tr22[ti], sample_weight=w_tr22[ti])
        p = model.predict_proba(X_tr22[vi])[:,1]
        aucs.append(roc_auc_score(y_tr22[vi], p, sample_weight=w_tr22[vi]))
    return (float(np.mean(aucs)),)

if "FitnessMax" not in dir(creator):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int",   random.randint, 0, 4)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=len(PNAMES))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate_ga)
toolbox.register("mate",       tools.cxTwoPoint)
toolbox.register("mutate",     tools.mutUniformInt, low=0, up=4, indpb=0.15)
toolbox.register("select",     tools.selTournament, tournsize=3)

random.seed(SEED); np.random.seed(SEED)
pop      = toolbox.population(n=GA_POP)
hof      = tools.HallOfFame(1)
stats_ga = tools.Statistics(lambda ind: ind.fitness.values)
stats_ga.register("max",  np.max)
stats_ga.register("mean", np.mean)

print(f"  Running GA: P={GA_POP} individuals × G={GA_NGEN} generations ...")
pop, logbook = algorithms.eaSimple(
    pop, toolbox, cxpb=0.6, mutpb=0.25, ngen=GA_NGEN,
    stats=stats_ga, halloffame=hof, verbose=False)

BEST_PARAMS  = decode_ga(hof[0])
BEST_CV_AUC  = hof[0].fitness.values[0]
gen_max_vals = [r["max"]  for r in logbook]
gen_mean_vals= [r["mean"] for r in logbook]

print(f"  Best 5-fold CV AUC: {BEST_CV_AUC:.4f}")
print(f"  Best params: {BEST_PARAMS}")


hdr("PART 4: Train All Models")

lr_pipe = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, C=0.1,
                     class_weight="balanced", random_state=SEED))])
lr_pipe.fit(X_tr22, y_tr22, lr__sample_weight=w_tr22/w_tr22.mean())
p_lr = lr_pipe.predict_proba(X_te22)[:,1]

rf = RandomForestClassifier(n_estimators=300, max_depth=6,
     class_weight="balanced", random_state=SEED, n_jobs=-1)
rf.fit(X_tr22, y_tr22, sample_weight=w_tr22)
p_rf = rf.predict_proba(X_te22)[:,1]

xgb_def = xgb.XGBClassifier(**DEFAULT_PARAMS, eval_metric="logloss",
           random_state=SEED, n_jobs=-1)
xgb_def.fit(X_tr22, y_tr22, sample_weight=w_tr22)
p_def = xgb_def.predict_proba(X_te22)[:,1]

xgb_ga = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
          random_state=SEED, n_jobs=-1)
xgb_ga.fit(X_tr22, y_tr22, sample_weight=w_tr22)
p_ga    = xgb_ga.predict_proba(X_te22)[:,1]
p_ga_tr = xgb_ga.predict_proba(X_tr22)[:,1]

lgb_m = lgb.LGBMClassifier(
    n_estimators=BEST_PARAMS.get("n_estimators",300),
    max_depth=BEST_PARAMS.get("max_depth",5),
    learning_rate=BEST_PARAMS.get("learning_rate",0.05),
    subsample=BEST_PARAMS.get("subsample",0.8),
    colsample_bytree=BEST_PARAMS.get("colsample_bytree",0.7),
    reg_alpha=BEST_PARAMS.get("reg_alpha",0.1),
    reg_lambda=BEST_PARAMS.get("reg_lambda",1.0),
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)
lgb_m.fit(X_tr22, y_tr22, sample_weight=w_tr22)
p_lgb = lgb_m.predict_proba(X_te22)[:,1]

t_lr  = opt_thresh(y_te22, p_lr,  w_te22)
t_rf  = opt_thresh(y_te22, p_rf,  w_te22)
t_def = opt_thresh(y_te22, p_def, w_te22)
t_ga  = opt_thresh(y_te22, p_ga,  w_te22)
t_lgb = opt_thresh(y_te22, p_lgb, w_te22)

yd_lr  = (p_lr  >= t_lr).astype(int)
yd_rf  = (p_rf  >= t_rf).astype(int)
yd_def = (p_def >= t_def).astype(int)
yd_ga  = (p_ga  >= t_ga).astype(int)
yd_lgb = (p_lgb >= t_lgb).astype(int)

null_brier = brier_score_loss(y_te22, np.full_like(p_ga, y_te22.mean()),
                               sample_weight=w_te22)
brier_ga   = brier_score_loss(y_te22, p_ga, sample_weight=w_te22)
brier_rf   = brier_score_loss(y_te22, p_rf, sample_weight=w_te22)

print(f"\n  GA-XGBoost Brier: {brier_ga:.4f}")
print(f"  Random Forest Brier: {brier_rf:.4f}")
print(f"  Null model Brier: {null_brier:.4f}")
print(f"  GA-XGBoost improvement over RF: {(brier_rf-brier_ga)/brier_rf*100:.1f}%")
print(f"  GA-XGBoost improvement over null: {(null_brier-brier_ga)/null_brier*100:.1f}%")


hdr("PART 4b: Multi-Outcome Evaluation")

multi_results = {}
for out, lbl in [("underweight", "Underweight (WAZ<−2)"),
                  ("wasted",      "Wasting (WHZ<−2)"),
                  ("any_malnut",  "Any malnutrition")]:
    if out not in df22.columns:
        continue
    yo    = df22[out].values.astype(float)
    valid = ~np.isnan(yo)
    if valid.sum() < 50 or yo[valid].sum() < 10:
        continue
    yo_c = yo[valid].astype(int)
    Xo   = X22[valid]
    wo   = w22[valid]
    Xtr_o, Xte_o, ytr_o, yte_o, wtr_o, wte_o = train_test_split(
        Xo, yo_c, wo, test_size=0.2, random_state=SEED, stratify=yo_c)
    m_out = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                               random_state=SEED, n_jobs=-1)
    m_out.fit(Xtr_o, ytr_o, sample_weight=wtr_o)
    yp_out  = m_out.predict_proba(Xte_o)[:,1]
    auc_out = roc_auc_score(yte_o, yp_out, sample_weight=wte_o)
    bs_out  = brier_score_loss(yte_o, yp_out, sample_weight=wte_o)
    prev_out= np.average(yo_c, weights=wo) * 100
    multi_results[out] = {
        "label": lbl, "auc": auc_out, "brier": bs_out,
        "prev": prev_out, "n": int(valid.sum()),
    }
    print(f"  {lbl:<35} prev={prev_out:.1f}%  AUC={auc_out:.4f}  Brier={bs_out:.4f}")

multi_df = pd.DataFrame([
    {"Outcome": v["label"], "Prevalence (%)": round(v["prev"],1),
     "n": f"{v['n']:,}", "AUC": round(v["auc"],4), "Brier": round(v["brier"],4)}
    for v in multi_results.values()
])
multi_df.to_csv(os.path.join(OUTPUT_DIR, "multi_outcome.csv"), index=False)


hdr("PART 4c: Sensitivity Analyses")

sensitivity_results = []

def run_sensitivity(label, mask):
    n_tot = int(mask.sum())
    if n_tot < 50:
        print(f"  {label:<40} skipped (n={n_tot})")
        return
    X_s = X22[mask]; y_s = y22[mask]; w_s = w22[mask]
    if y_s.sum() < 20 or (1 - y_s).sum() < 20:
        print(f"  {label:<40} skipped (insufficient positives/negatives)")
        return
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X_s, y_s, w_s, test_size=0.2, random_state=SEED, stratify=y_s)
    m_s = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1)
    m_s.fit(Xtr, ytr, sample_weight=wtr)
    yp_s  = m_s.predict_proba(Xte)[:,1]
    auc_s = roc_auc_score(yte, yp_s, sample_weight=wte)
    bs_s  = brier_score_loss(yte, yp_s, sample_weight=wte)
    prev_s= np.average(y_s, weights=w_s) * 100
    print(f"  {label:<40} n={n_tot:,}  prev={prev_s:.1f}%  AUC={auc_s:.4f}  Brier={bs_s:.4f}")
    sensitivity_results.append({
        "Analysis": label, "n": n_tot,
        "Prevalence (%)": round(prev_s, 1),
        "AUC": round(auc_s, 4),
        "Brier": round(bs_s, 4),
    })

cc_mask = df22[FEATS22].notna().all(axis=1).values
run_sensitivity("Complete-case only", cc_mask)

age6_mask = (df22["child_age_mo"] >= 6).values
run_sensitivity("Age >= 6 months", age6_mask)

if "urban" in df22.columns:
    run_sensitivity("Urban only", (df22["urban"] == 1).values)
    run_sensitivity("Rural only", (df22["urban"] == 0).values)

df22["mod_stunted"] = (df22["HW70"] < -150).astype(int)
y22_mod = df22["mod_stunted"].values.astype(int)
X_mod   = X22.copy()
Xtr_m, Xte_m, ytr_m, yte_m, wtr_m, wte_m = train_test_split(
    X_mod, y22_mod, w22, test_size=0.2, random_state=SEED, stratify=y22_mod)
m_mod = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                            random_state=SEED, n_jobs=-1)
m_mod.fit(Xtr_m, ytr_m, sample_weight=wtr_m)
yp_mod  = m_mod.predict_proba(Xte_m)[:,1]
auc_mod = roc_auc_score(yte_m, yp_mod, sample_weight=wte_m)
bs_mod  = brier_score_loss(yte_m, yp_mod, sample_weight=wte_m)
prev_mod= np.average(y22_mod, weights=w22) * 100
print(f"  {'HAZ < -1.5 SD (moderate threshold)':<40} n={len(df22):,}  "
      f"prev={prev_mod:.1f}%  AUC={auc_mod:.4f}  Brier={bs_mod:.4f}")
sensitivity_results.append({
    "Analysis": "HAZ < -1.5 SD (moderate)", "n": len(df22),
    "Prevalence (%)": round(prev_mod, 1),
    "AUC": round(auc_mod, 4), "Brier": round(bs_mod, 4),
})

run_sensitivity("Full cohort (reference)", np.ones(len(df22), dtype=bool))

sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_analyses.csv"), index=False)


hdr("PART 4d: Subgroup Performance")

subgroup_results = []

for d in sorted(df22["division"].dropna().unique()):
    mask = (div_te22 == d)
    if mask.sum() < 10 or y_te22[mask].sum() < 2: continue
    auc_sg = roc_auc_score(y_te22[mask], p_ga[mask], sample_weight=w_te22[mask])
    subgroup_results.append({
        "Subgroup": f"Division: {DIV_LABELS.get(int(d), int(d))}",
        "n": int(mask.sum()), "AUC": round(auc_sg, 4),
    })

for q in range(1, 6):
    mask = (wl_te22 == q)
    if mask.sum() < 10 or y_te22[mask].sum() < 2: continue
    auc_sg = roc_auc_score(y_te22[mask], p_ga[mask], sample_weight=w_te22[mask])
    subgroup_results.append({
        "Subgroup": f"Wealth Q{q}",
        "n": int(mask.sum()), "AUC": round(auc_sg, 4),
    })

if "age_group" in FEATS22:
    age_te_vals = X_te22[:, FEATS22.index("age_group")]
    for ag, (lo_a, hi_a) in enumerate([(0,5),(6,11),(12,23),(24,35),(36,47),(48,59)]):
        mask = (age_te_vals == ag)
        if mask.sum() < 10 or y_te22[mask].sum() < 2: continue
        auc_sg = roc_auc_score(y_te22[mask], p_ga[mask], sample_weight=w_te22[mask])
        subgroup_results.append({
            "Subgroup": f"Age {lo_a}–{hi_a}m",
            "n": int(mask.sum()), "AUC": round(auc_sg, 4),
        })

sg_df = pd.DataFrame(subgroup_results)
sg_df.to_csv(os.path.join(OUTPUT_DIR, "subgroup_performance.csv"), index=False)
print(sg_df.to_string(index=False))


hdr("PART 5: Bootstrap Evaluation (500-rep PSU-cluster)")

MODELS_DICT = {
    "Logistic Regression":    (p_lr,  yd_lr),
    "Random Forest":          (p_rf,  yd_rf),
    "XGBoost (default)":      (p_def, yd_def),
    "XGBoost (GA-optimised)": (p_ga,  yd_ga),
    "LightGBM":               (p_lgb, yd_lgb),
}
boot_results = {}
for name, (yp, yd) in MODELS_DICT.items():
    print(f"  Bootstrapping {name} ...", end=" ", flush=True)
    boot_results[name] = bootstrap_metrics(
        y_te22, yp, yd, w_te22, psu_te22, str_te22, n_boot=N_BOOT)
    a, lo, hi = boot_results[name]["auc"]
    b         = boot_results[name]["brier"][0]
    print(f"AUC={a:.4f} [{lo:.4f}–{hi:.4f}]  Brier={b:.4f}")


hdr("PART 6: SHAP — Global Importance + Cross-Fold Stability")

explainer = shap.TreeExplainer(xgb_ga)
shap_te   = explainer.shap_values(X_te22)
mean_shap = np.abs(shap_te).mean(axis=0)
feat_imp  = pd.Series(mean_shap, index=FEATS22).sort_values(ascending=False)

fold_shap_means = []
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold_i, (tr_idx, val_idx) in enumerate(cv5.split(X_tr22, y_tr22)):
    m_fold = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1)
    m_fold.fit(X_tr22[tr_idx], y_tr22[tr_idx], sample_weight=w_tr22[tr_idx])
    sv_fold = shap.TreeExplainer(m_fold).shap_values(X_tr22[val_idx])
    fold_shap_means.append(
        pd.Series(np.abs(sv_fold).mean(axis=0), index=FEATS22))
    print(f"  Fold {fold_i+1}/5 done")

top10_feats = feat_imp.head(10).index.tolist()
shap_stab   = pd.DataFrame(
    {f"Fold{i+1}": fs[top10_feats] for i, fs in enumerate(fold_shap_means)})
shap_stab["mean"] = shap_stab.mean(axis=1)
shap_stab["sd"]   = shap_stab[[f"Fold{i+1}" for i in range(5)]].std(axis=1)
shap_stab["cv"]   = shap_stab["sd"] / shap_stab["mean"] * 100
shap_stab.to_csv(os.path.join(OUTPUT_DIR, "shap_stability.csv"))

print(f"\n  Top 10 predictors:")
for i, (f, v) in enumerate(feat_imp.head(10).items(), 1):
    cv_val = shap_stab.loc[f,"cv"] if f in shap_stab.index else np.nan
    print(f"  {i:2d}. {f:<30} SHAP={v:.4f}  CV={cv_val:.1f}%")


hdr("PART 7: Fairness Analysis")

s_bin = (wl_te22 <= 2).astype(int)

yd_fair, group_thresh = apply_fairness_calibration(
    p_ga_tr, y_tr22, w_tr22, wl_tr22,
    p_ga,    wl_te22)

p_def_tr = xgb_def.predict_proba(X_tr22)[:,1]
yd_def_fair, _ = apply_fairness_calibration(
    p_def_tr, y_tr22, w_tr22, wl_tr22,
    p_def,    wl_te22)

fair_models = {
    "XGB Default":            yd_def,
    "XGB Default + Fairness": yd_def_fair,
    "XGB GA":                 yd_ga,
    "XGB GA + Fairness":      yd_fair,
}
fair_table = {}
print(f"\n  {'Model':<28} {'|DPD|':>8}  {'|EOD|':>8}  {'AUC':>8}")
print(f"  {'-'*55}")
for name, ypred in fair_models.items():
    dpd = demographic_parity_difference(y_te22, ypred, sensitive_features=s_bin)
    eod = equalized_odds_difference(y_te22, ypred, sensitive_features=s_bin)
    auc = roc_auc_score(y_te22, p_ga if "GA" in name else p_def,
                        sample_weight=w_te22)
    fair_table[name] = {"dpd": dpd, "eod": eod, "auc": auc}
    print(f"  {name:<28} {abs(dpd):>8.4f}  {abs(eod):>8.4f}  {auc:>8.4f}")

fnr_by_quintile = {}
for mname, ypred in fair_models.items():
    fnrs = []
    for q in range(1, 6):
        m = (wl_te22==q) & (y_te22==1)
        fnrs.append((1 - np.average(ypred[m], weights=w_te22[m]))*100
                    if m.sum()>0 else np.nan)
    fnr_by_quintile[mname] = fnrs

print(f"\n  Q3 Subgroup TPR Analysis:")
for mname, ypred in {"XGB GA (unfair)": yd_ga,
                      "XGB GA + Fairness": yd_fair}.items():
    m3  = (wl_te22==3) & (y_te22==1)
    m3_all = (wl_te22==3)
    n3  = m3.sum()
    n3_all = m3_all.sum()
    if n3 > 0:
        tpr3 = np.average(ypred[m3], weights=w_te22[m3]) * 100
        auc3 = roc_auc_score(y_te22[m3_all], p_ga[m3_all]) \
               if m3_all.sum() > 10 and y_te22[m3_all].sum() > 1 else np.nan
        print(f"  {mname:<30} Q3 TPR={tpr3:.1f}%  "
              f"n_pos={n3}  n_total={n3_all}  AUC={auc3:.3f}")

extended_fair = {}
if "urban" in FEATS22:
    urban_te = X_te22[:, FEATS22.index("urban")].astype(int)
    extended_fair["Urban vs Rural"] = {
        "DPD": abs(demographic_parity_difference(y_te22, yd_fair, sensitive_features=urban_te)),
        "EOD": abs(equalized_odds_difference(y_te22, yd_fair, sensitive_features=urban_te)),
    }
if "mat_no_edu" in FEATS22:
    edu_te = X_te22[:, FEATS22.index("mat_no_edu")].astype(int)
    extended_fair["No Education vs Any"] = {
        "DPD": abs(demographic_parity_difference(y_te22, yd_fair, sensitive_features=edu_te)),
        "EOD": abs(equalized_odds_difference(y_te22, yd_fair, sensitive_features=edu_te)),
    }
poorest_te = (wl_te22==1).astype(int)
extended_fair["Poorest vs Others"] = {
    "DPD": abs(demographic_parity_difference(y_te22, yd_fair, sensitive_features=poorest_te)),
    "EOD": abs(equalized_odds_difference(y_te22, yd_fair, sensitive_features=poorest_te)),
}

print(f"\n  Extended fairness (calibrated model):")
for grp, vals in extended_fair.items():
    print(f"    {grp:<35} |DPD|={vals['DPD']:.4f}  |EOD|={vals['EOD']:.4f}")


if RUN_ABLATION:
    hdr("PART 8: Ablation Study")

    mat_anthro_feats = ["mat_height_cm","mat_bmi","mat_short","mat_underweight",
                         "mat_overweight","mat_age","mat_teen","mat_height_sq"]

    ablation_configs = {
        "Full SAFE-XAI":           {"sw":True, "ma":True, "ga":True,  "fair":True},
        "– Survey weights":        {"sw":False,"ma":True, "ga":True,  "fair":True},
        "– Maternal anthropometry":{"sw":True, "ma":False,"ga":True,  "fair":True},
        "– GA optimisation":       {"sw":True, "ma":True, "ga":False, "fair":True},
        "– Fairness calibration":  {"sw":True, "ma":True, "ga":True,  "fair":False},
        "Default XGB + Fairness":  {"sw":True, "ma":True, "ga":False, "fair":True},
    }

    ablation_results = []
    for config_name, cfg in ablation_configs.items():
        print(f"\n  Config: {config_name}")

        feat_use = [f for f in FEATS22
                    if not (not cfg["ma"] and f in mat_anthro_feats)]
        fidx     = [FEATS22.index(f) for f in feat_use]
        Xtr_abl  = X_tr22[:, fidx]
        Xte_abl  = X_te22[:, fidx]
        sw_use   = w_tr22 if cfg["sw"] else np.ones(len(w_tr22))
        par_use  = BEST_PARAMS if cfg["ga"] else DEFAULT_PARAMS

        m_abl = xgb.XGBClassifier(**par_use, eval_metric="logloss",
                                    random_state=SEED, n_jobs=-1)
        m_abl.fit(Xtr_abl, y_tr22, sample_weight=sw_use)
        p_abl    = m_abl.predict_proba(Xte_abl)[:,1]
        p_abl_tr = m_abl.predict_proba(Xtr_abl)[:,1]
        t_abl    = opt_thresh(y_te22, p_abl, w_te22)
        yd_abl   = (p_abl >= t_abl).astype(int)
        auc_abl  = roc_auc_score(y_te22, p_abl, sample_weight=w_te22)

        if cfg["fair"]:
            yd_fair_abl, _ = apply_fairness_calibration(
                p_abl_tr, y_tr22, sw_use if cfg["sw"] else w_tr22,
                wl_tr22, p_abl, wl_te22)
        else:
            yd_fair_abl = yd_abl

        eod_abl = abs(equalized_odds_difference(
            y_te22, yd_fair_abl, sensitive_features=s_bin))
        dpd_abl = abs(demographic_parity_difference(
            y_te22, yd_fair_abl, sensitive_features=s_bin))

        br_abl = bootstrap_metrics(
            y_te22, p_abl, yd_abl, w_te22, psu_te22, str_te22, n_boot=200)
        auc_lo, auc_hi = br_abl["auc"][1], br_abl["auc"][2]

        print(f"    AUC={auc_abl:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]  "
              f"|EOD|={eod_abl:.4f}  |DPD|={dpd_abl:.4f}  n_feats={len(feat_use)}")

        ablation_results.append({
            "Configuration": config_name,
            "n_features": len(feat_use),
            "AUC": round(auc_abl, 4),
            "AUC_lo": round(auc_lo, 4),
            "AUC_hi": round(auc_hi, 4),
            "EOD": round(eod_abl, 4),
            "DPD": round(dpd_abl, 4),
        })

    abl_df = pd.DataFrame(ablation_results)
    abl_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)

    full_auc     = abl_df.loc[abl_df["Configuration"]=="Full SAFE-XAI", "AUC"].values[0]
    def_fair_auc = abl_df.loc[abl_df["Configuration"]=="Default XGB + Fairness", "AUC"].values[0]
    full_eod     = abl_df.loc[abl_df["Configuration"]=="Full SAFE-XAI", "EOD"].values[0]
    def_fair_eod = abl_df.loc[abl_df["Configuration"]=="Default XGB + Fairness", "EOD"].values[0]
    print(f"\n  GA adds {full_auc-def_fair_auc:+.4f} AUC vs Default+Fair")
    print(f"  GA changes EOD by {full_eod-def_fair_eod:+.4f} vs Default+Fair")


hdr("PART 9: Spatial Analysis — LOO-CV District Atlas & Moran's I")

X_full22   = X22.copy()
y_full22   = df22["stunted"].values.astype(int)
w_full22   = df22["survey_weight"].values
clus_ids22 = df22["cluster_id"].values

oof_probs = np.zeros(len(df22))
for clus in np.unique(clus_ids22):
    te_m = clus_ids22 == clus
    tr_m = ~te_m
    if tr_m.sum() < 50 or te_m.sum() < 1: continue
    m_tmp = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                               random_state=SEED, n_jobs=-1)
    m_tmp.fit(X_full22[tr_m], y_full22[tr_m], sample_weight=w_full22[tr_m])
    oof_probs[te_m] = m_tmp.predict_proba(X_full22[te_m])[:,1]

df22 = df22.copy()
df22["oof_pred_prob"] = oof_probs

dist_stats = pd.DataFrame()
if "district" in df22.columns and df22["district"].notna().sum() > 100:
    dist_stats = df22.groupby("district").apply(lambda g: pd.Series({
        "n_children":         len(g),
        "observed_stunting":  np.average(g["stunted"], weights=g["survey_weight"])*100,
        "predicted_stunting": np.average(g["oof_pred_prob"], weights=g["survey_weight"])*100,
        "underweight":        np.average(g["underweight"].fillna(0), weights=g["survey_weight"])*100,
        "wasted":             np.average(g["wasted"].fillna(0), weights=g["survey_weight"])*100,
        "pct_poor":           np.average((g["V190"]<=2).astype(float), weights=g["survey_weight"])*100,
        "mean_wealth":        np.average(g["wealth_idx"], weights=g["survey_weight"]),
        "mat_height_mean":    g["mat_height_cm"].mean(),
        "WASH_mean":          np.average(g["WASH_score"], weights=g["survey_weight"]),
        "n_clusters":         g["cluster_id"].nunique(),
    })).reset_index()
    dist_stats["risk_tier"] = pd.cut(
        dist_stats["predicted_stunting"],
        bins=[0,15,22,30,100],
        labels=["Low (<15%)","Moderate (15-22%)","High (22-30%)","Very High (>30%)"])

    cluster_stats = df22.groupby("cluster_id").apply(lambda g: pd.Series({
        "residual": np.average(g["stunted"]-g["oof_pred_prob"],
                               weights=g["survey_weight"])*100,
        "district": g["district"].mode()[0] if len(g)>0 else np.nan,
    })).reset_index()
    cluster_stats["spatial_lag"] = cluster_stats.groupby("district")["residual"].transform("mean")
    r_moran, p_moran = stats.pearsonr(
        cluster_stats["residual"],
        cluster_stats["spatial_lag"] - cluster_stats["spatial_lag"].mean())
    r_dist, p_dist   = stats.pearsonr(
        dist_stats["observed_stunting"], dist_stats["predicted_stunting"])
    print(f"  Moran's I: r={r_moran:.3f}  p={p_moran:.4f}")
    print(f"  District obs vs pred: r={r_dist:.3f}  p={p_dist:.4f}")
    dist_stats.to_csv(os.path.join(OUTPUT_DIR, "district_atlas.csv"), index=False)

if RUN_SPATIAL_SENS and "district" in df22.columns and df22["district"].notna().sum() > 100:
    sub("Spatial Sensitivity: District Fixed Effects")
    dist_dummies = pd.get_dummies(df22["district"].fillna(-1).astype(int),
                                   prefix="dist").values.astype(float)
    X_spatial_full = np.hstack([X_full22, dist_dummies])

    (X_sp_tr, X_sp_te,
     y_sp_tr, y_sp_te,
     w_sp_tr, w_sp_te) = train_test_split(
        X_spatial_full, y_full22, w_full22,
        test_size=0.20, random_state=SEED, stratify=y_full22)

    m_spatial = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                                    random_state=SEED, n_jobs=-1)
    m_spatial.fit(X_sp_tr, y_sp_tr, sample_weight=w_sp_tr)
    p_spatial = m_spatial.predict_proba(X_sp_te)[:,1]
    auc_spatial = roc_auc_score(y_sp_te, p_spatial, sample_weight=w_sp_te)
    auc_base    = roc_auc_score(y_sp_te, xgb_ga.predict_proba(
        X_spatial_full[:len(X_sp_te), :len(FEATS22)])[:,1], sample_weight=w_sp_te)

    print(f"  Without district FE: AUC = {auc_base:.4f}")
    print(f"  With district FE:    AUC = {auc_spatial:.4f}")
    print(f"  Delta AUC:           {auc_spatial-auc_base:+.4f}")


hdr("PART 10: External Temporal Validation — BDHS 2017 (frozen model)")

df17    = round_dfs[2017]
FEATS17 = round_feats[2017]

missing_in17 = [f for f in FEATS22 if f not in FEATS17]
common_17    = [f for f in FEATS22 if f in FEATS17]
print(f"  Features in 2022: {len(FEATS22)}")
print(f"  Features in 2017: {len(FEATS17)}")
print(f"  Common features:  {len(common_17)}")

if missing_in17:
    print(f"  Features absent in 2017 (set to NaN, then imputed):")
    for f in missing_in17:
        shap_rank = list(feat_imp.index).index(f)+1 if f in feat_imp.index else "unranked"
        print(f"    {f:<35} SHAP rank={shap_rank}")

X17_raw = np.zeros((len(df17), len(FEATS22)))
for i, f in enumerate(FEATS22):
    if f in common_17 and f in df17.columns:
        X17_raw[:, i] = df17[f].values
    else:
        X17_raw[:, i] = np.nan

med_imp = SimpleImputer(strategy="median")
med_imp.fit(X_tr22)
X17 = med_imp.transform(X17_raw)

y17  = df17["stunted"].values.astype(int)
w17  = df17["survey_weight"].values
psu17 = df17["psu"].values
str17 = df17["strata"].values
wl17  = df17["V190"].fillna(3).values.astype(int)

p17  = xgb_ga.predict_proba(X17)[:,1]
t17  = opt_thresh(y17, p17, w17)
yd17 = (p17 >= t17).astype(int)

auc17    = roc_auc_score(y17, p17, sample_weight=w17)
brier17  = brier_score_loss(y17, p17, sample_weight=w17)
prev17   = np.average(y17, weights=w17) * 100

from sklearn.linear_model import LogisticRegression as LR_cal
log_odds17 = np.log(p17.clip(1e-6, 1-1e-6) / (1-p17.clip(1e-6, 1-1e-6)))
cal_model  = LR_cal(fit_intercept=True)
cal_model.fit(log_odds17.reshape(-1,1), y17, sample_weight=w17)
cal_slope17    = cal_model.coef_[0][0]
cal_intercept17 = cal_model.intercept_[0]

br17 = bootstrap_metrics(y17, p17, yd17, w17, psu17, str17, n_boot=N_BOOT)
auc17_lo, auc17_hi = br17["auc"][1], br17["auc"][2]

rng_platt  = np.random.default_rng(SEED)
cal_idx    = rng_platt.choice(len(y17), size=len(y17)//2, replace=False)
eval_idx17 = np.setdiff1d(np.arange(len(y17)), cal_idx)
log_odds_cal = np.log(p17[cal_idx].clip(1e-6,1-1e-6) /
                       (1-p17[cal_idx].clip(1e-6,1-1e-6)))
platt_m = LR_cal(fit_intercept=True)
platt_m.fit(log_odds_cal.reshape(-1,1), y17[cal_idx], sample_weight=w17[cal_idx])
log_odds_eval17 = np.log(p17[eval_idx17].clip(1e-6,1-1e-6) /
                          (1-p17[eval_idx17].clip(1e-6,1-1e-6)))
p17_recal      = platt_m.predict_proba(log_odds_eval17.reshape(-1,1))[:,1]
auc17_recal    = roc_auc_score(y17[eval_idx17], p17_recal, sample_weight=w17[eval_idx17])
brier17_recal  = brier_score_loss(y17[eval_idx17], p17_recal, sample_weight=w17[eval_idx17])
brier17_uncal  = brier_score_loss(y17[eval_idx17], p17[eval_idx17], sample_weight=w17[eval_idx17])

s17_bin = (wl17 <= 2).astype(int)
yd17_fair, _ = apply_fairness_calibration(
    p_ga_tr, y_tr22, w_tr22, wl_tr22, p17, wl17)
dpd17 = abs(demographic_parity_difference(y17, yd17, sensitive_features=s17_bin))
eod17 = abs(equalized_odds_difference(y17, yd17, sensitive_features=s17_bin))

print(f"\n  Development (BDHS 2022): AUC={boot_results['XGBoost (GA-optimised)']['auc'][0]:.4f}")
print(f"  Validation  (BDHS 2017): AUC={auc17:.4f} [{auc17_lo:.4f}–{auc17_hi:.4f}]")
print(f"  AUC drop:                {boot_results['XGBoost (GA-optimised)']['auc'][0]-auc17:+.4f}")
print(f"  Calibration slope:       {cal_slope17:.4f}")
print(f"  Calibration intercept:   {cal_intercept17:.4f}")
print(f"  Brier (uncalibrated):    {brier17:.4f}")
print(f"  Brier (recalibrated):    {brier17_recal:.4f}")
print(f"  |DPD|: {dpd17:.4f}  |EOD|: {eod17:.4f}")


if RUN_PANEL_ANALYSIS:
    hdr("PART 11: Panel Analysis — BDHS 2011–2022")

    sub("P1: Rolling Temporal Validation")
    rolling_pairs = [
        (2011, 2014),
        (2014, 2017),
        (2017, 2022),
        ([2011,2014], 2017),
        ([2011,2014,2017], 2022),
    ]
    rolling_results = []

    for train_yrs, val_yr in rolling_pairs:
        train_yrs_list = [train_yrs] if isinstance(train_yrs, int) else train_yrs
        label = f"Train {'+'.join(map(str,train_yrs_list))} → Val {val_yr}"

        dfs_train = [round_dfs[y] for y in train_yrs_list]
        df_tr_roll = pd.concat(dfs_train, ignore_index=True, sort=False)

        Xtr_list = []
        for f in FEATS22:
            if f in df_tr_roll.columns:
                Xtr_list.append(df_tr_roll[f].values)
            else:
                Xtr_list.append(np.full(len(df_tr_roll), np.nan))
        Xtr_r = np.column_stack(Xtr_list)

        ytr_r = df_tr_roll["stunted"].values.astype(int)
        wtr_r = df_tr_roll["survey_weight"].values

        imp_r = SimpleImputer(strategy="median")
        Xtr_r_imp = imp_r.fit_transform(Xtr_r)

        df_val_roll = round_dfs[val_yr]
        Xval_list = []
        for f in FEATS22:
            if f in df_val_roll.columns:
                Xval_list.append(df_val_roll[f].values)
            else:
                Xval_list.append(np.full(len(df_val_roll), np.nan))
        Xval_r     = np.column_stack(Xval_list)
        Xval_r_imp = imp_r.transform(Xval_r)

        yval_r = df_val_roll["stunted"].values.astype(int)
        wval_r = df_val_roll["survey_weight"].values

        m_roll = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                                     random_state=SEED, n_jobs=-1)
        m_roll.fit(Xtr_r_imp, ytr_r, sample_weight=wtr_r)
        p_roll    = m_roll.predict_proba(Xval_r_imp)[:,1]
        auc_roll  = roc_auc_score(yval_r, p_roll, sample_weight=wval_r)
        brier_roll= brier_score_loss(yval_r, p_roll, sample_weight=wval_r)
        prev_val  = np.average(yval_r, weights=wval_r) * 100
        prev_tr   = np.average(ytr_r, weights=wtr_r) * 100

        exp_roll  = shap.TreeExplainer(m_roll)
        sv_roll   = exp_roll.shap_values(Xval_r_imp[:500])
        shap_roll = safe_shap_series(sv_roll, FEATS22, label=label)
        mat_ht_rank = shap_roll.rank(ascending=False)[
            "mat_height_cm"] if "mat_height_cm" in shap_roll.index else np.nan
        mat_ht_shap = shap_roll["mat_height_cm"] if "mat_height_cm" in shap_roll.index else np.nan

        print(f"  {label}")
        print(f"    AUC={auc_roll:.4f}  Brier={brier_roll:.4f}  "
              f"prev_train={prev_tr:.1f}%  prev_val={prev_val:.1f}%")
        print(f"    mat_height_cm SHAP rank={mat_ht_rank:.0f}  |SHAP|={mat_ht_shap:.4f}")

        rolling_results.append({
            "Training": "+".join(map(str, train_yrs_list)),
            "Validation": val_yr,
            "n_train": len(ytr_r),
            "n_val":   len(yval_r),
            "prev_train": round(prev_tr, 1),
            "prev_val":   round(prev_val, 1),
            "AUC":        round(auc_roll, 4),
            "Brier":      round(brier_roll, 4),
            "mat_height_shap_rank": mat_ht_rank,
            "mat_height_shap": round(mat_ht_shap, 4),
        })

    roll_df = pd.DataFrame(rolling_results)
    roll_df.to_csv(os.path.join(OUTPUT_DIR, "rolling_validation.csv"), index=False)

    sub("P2: Cross-Round SHAP Stability — Maternal Height Dominance")
    round_shap_ranks = {}
    round_shap_vals  = {}
    for yr in sorted(round_dfs.keys()):
        df_yr  = round_dfs[yr]
        Xyr_list = []
        for f in FEATS22:
            if f in df_yr.columns:
                Xyr_list.append(df_yr[f].values)
            else:
                Xyr_list.append(np.full(len(df_yr), np.nan))
        Xyr = np.column_stack(Xyr_list)
        yyr = df_yr["stunted"].values.astype(int)
        wyr = df_yr["survey_weight"].values

        imp_yr  = SimpleImputer(strategy="median")
        Xyr_imp = imp_yr.fit_transform(Xyr)

        m_yr = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                                   random_state=SEED, n_jobs=-1)
        m_yr.fit(Xyr_imp, yyr, sample_weight=wyr)

        n_shap = min(1000, len(Xyr_imp))
        sv_yr  = shap.TreeExplainer(m_yr).shap_values(Xyr_imp[:n_shap])
        si_yr  = safe_shap_series(sv_yr, FEATS22, label=str(yr)).sort_values(ascending=False)
        round_shap_ranks[yr] = {f: int(si_yr.rank(ascending=False)[f])
                                 for f in si_yr.index}
        round_shap_vals[yr]  = si_yr

        top3 = si_yr.head(3)
        ht_rank = round_shap_ranks[yr].get("mat_height_cm","?")
        print(f"  {yr}: mat_height rank={ht_rank}  top3={list(top3.index)}")

    shap_cross = pd.DataFrame(
        {yr: round_shap_vals[yr][top10_feats] for yr in sorted(round_dfs.keys())})
    shap_cross["mean_across_rounds"] = shap_cross.mean(axis=1)
    shap_cross["cv_across_rounds"]   = shap_cross[[y for y in sorted(round_dfs)]].std(axis=1) / \
                                        shap_cross[[y for y in sorted(round_dfs)]].mean(axis=1) * 100
    shap_cross.to_csv(os.path.join(OUTPUT_DIR, "shap_cross_round.csv"))

    sub("P3: Covariate Drift — Feature Distribution Shift 2011→2022")
    drift_results = []
    drift_feats   = ["mat_height_cm","mat_bmi","wealth_idx","anc_visits",
                      "WASH_score","mat_edu_yrs","improved_sanit","urban"]
    for f in drift_feats:
        vals = {}
        for yr in sorted(round_dfs.keys()):
            if f in round_dfs[yr].columns:
                w_yr = round_dfs[yr]["survey_weight"]
                v_yr = round_dfs[yr][f].dropna()
                w_yr = w_yr[v_yr.index]
                vals[yr] = np.average(v_yr, weights=w_yr)
        if len(vals) >= 2:
            yrs = sorted(vals.keys())
            trend = (vals[yrs[-1]] - vals[yrs[0]]) / vals[yrs[0]] * 100
            drift_results.append({"Feature": f,
                                   **{str(y): round(v,3) for y,v in vals.items()},
                                   "Trend (%)": round(trend, 1)})
            print(f"  {f:<30} " + "  ".join([f"{yr}:{v:.3f}" for yr,v in vals.items()]) +
                  f"  trend={trend:+.1f}%")

    drift_df = pd.DataFrame(drift_results)
    drift_df.to_csv(os.path.join(OUTPUT_DIR, "covariate_drift.csv"), index=False)

    sub("P4: Stunting Prevalence Trajectory 2011→2022")
    traj_rows = []
    for yr in sorted(round_dfs.keys()):
        df_yr = round_dfs[yr]
        w_yr  = df_yr["survey_weight"]
        for outcome, col in [("Stunting","stunted"),("Underweight","underweight"),
                              ("Wasting","wasted")]:
            if col in df_yr.columns:
                vals = df_yr[col].dropna()
                ww   = w_yr[vals.index]
                prev = np.average(vals, weights=ww) * 100
                traj_rows.append({"Year": yr, "Outcome": outcome,
                                   "Prevalence": round(prev, 1),
                                   "n": int(vals.sum())})
    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(os.path.join(OUTPUT_DIR, "prevalence_trajectory.csv"), index=False)
    print(traj_df.pivot(index="Outcome", columns="Year", values="Prevalence").to_string())

    sub("P5: Pooled Panel Model (all rounds, survey_year as feature)")
    Xpanel_list = []
    for f in FEATS22:
        col_vals = []
        for yr in sorted(round_dfs.keys()):
            if f in round_dfs[yr].columns:
                col_vals.append(round_dfs[yr][f].values)
            else:
                col_vals.append(np.full(len(round_dfs[yr]), np.nan))
        Xpanel_list.append(np.concatenate(col_vals))

    year_vals = np.concatenate([
        np.full(len(round_dfs[yr]), yr) for yr in sorted(round_dfs.keys())])
    Xpanel_list.append(year_vals)
    panel_feat_names = FEATS22 + ["survey_year"]

    Xpanel = np.column_stack(Xpanel_list)
    ypanel = np.concatenate([round_dfs[yr]["stunted"].values for yr in sorted(round_dfs)])
    wpanel = np.concatenate([round_dfs[yr]["survey_weight"].values for yr in sorted(round_dfs)])

    imp_panel  = SimpleImputer(strategy="median")
    Xpanel_imp = imp_panel.fit_transform(Xpanel)

    yr_arr     = year_vals.astype(int)
    tr_mask    = yr_arr < 2022
    val_mask   = yr_arr == 2022

    m_panel = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                                  random_state=SEED, n_jobs=-1)
    m_panel.fit(Xpanel_imp[tr_mask], ypanel[tr_mask],
                sample_weight=wpanel[tr_mask])
    p_panel = m_panel.predict_proba(Xpanel_imp[val_mask])[:,1]
    auc_panel = roc_auc_score(ypanel[val_mask], p_panel,
                               sample_weight=wpanel[val_mask])
    brier_panel = brier_score_loss(ypanel[val_mask], p_panel,
                                    sample_weight=wpanel[val_mask])

    sv_panel = shap.TreeExplainer(m_panel).shap_values(Xpanel_imp[val_mask][:500])
    si_panel = safe_shap_series(sv_panel, panel_feat_names, label="panel").sort_values(ascending=False)
    mat_ht_rank_panel = int(si_panel.rank(ascending=False)["mat_height_cm"]) \
                         if "mat_height_cm" in si_panel.index else np.nan

    print(f"\n  Panel model (train 2011+2014+2017 → validate 2022):")
    print(f"  AUC={auc_panel:.4f}  Brier={brier_panel:.4f}")
    print(f"  mat_height_cm SHAP rank={mat_ht_rank_panel}  |SHAP|={si_panel.get('mat_height_cm',0):.4f}")
    print(f"  survey_year SHAP rank={int(si_panel.rank(ascending=False)['survey_year'])}  "
          f"|SHAP|={si_panel.get('survey_year',0):.4f}")
    print(f"  Top 5 panel predictors: {list(si_panel.head(5).index)}")

    si_panel.to_csv(os.path.join(OUTPUT_DIR, "shap_panel_model.csv"))

    sub("P6: Per-Round Fairness — Wealth Detection Gap Over Time")
    round_fair_rows = []
    for yr in sorted(round_dfs.keys()):
        df_yr  = round_dfs[yr]
        Xyr_list = []
        for f in FEATS22:
            if f in df_yr.columns:
                Xyr_list.append(df_yr[f].values)
            else:
                Xyr_list.append(np.full(len(df_yr), np.nan))
        Xyr = np.column_stack(Xyr_list)
        yyr = df_yr["stunted"].values.astype(int)
        wyr = df_yr["survey_weight"].values
        wlyr= df_yr["V190"].fillna(3).values.astype(int)

        imp_yr  = SimpleImputer(strategy="median")
        Xyr_imp = imp_yr.fit_transform(Xyr)

        (Xtr_yr,Xte_yr,ytr_yr,yte_yr,
         wtr_yr,wte_yr,wltr_yr,wlte_yr) = train_test_split(
            Xyr_imp, yyr, wyr, wlyr,
            test_size=0.20, random_state=SEED, stratify=yyr)

        m_yr = xgb.XGBClassifier(**BEST_PARAMS, eval_metric="logloss",
                                   random_state=SEED, n_jobs=-1)
        m_yr.fit(Xtr_yr, ytr_yr, sample_weight=wtr_yr)
        p_yr     = m_yr.predict_proba(Xte_yr)[:,1]
        p_yr_tr  = m_yr.predict_proba(Xtr_yr)[:,1]
        yd_yr    = (p_yr >= opt_thresh(yte_yr, p_yr, wte_yr)).astype(int)
        yd_fair_yr, _ = apply_fairness_calibration(
            p_yr_tr, ytr_yr, wtr_yr, wltr_yr, p_yr, wlte_yr)

        s_yr = (wlte_yr <= 2).astype(int)
        dpd_yr_raw  = abs(demographic_parity_difference(yte_yr, yd_yr, sensitive_features=s_yr))
        eod_yr_raw  = abs(equalized_odds_difference(yte_yr, yd_yr, sensitive_features=s_yr))
        dpd_yr_fair = abs(demographic_parity_difference(yte_yr, yd_fair_yr, sensitive_features=s_yr))
        eod_yr_fair = abs(equalized_odds_difference(yte_yr, yd_fair_yr, sensitive_features=s_yr))
        auc_yr      = roc_auc_score(yte_yr, p_yr, sample_weight=wte_yr)

        print(f"  {yr}: AUC={auc_yr:.4f}  "
              f"|DPD|_raw={dpd_yr_raw:.4f}  |EOD|_raw={eod_yr_raw:.4f}  "
              f"|DPD|_fair={dpd_yr_fair:.4f}  |EOD|_fair={eod_yr_fair:.4f}")
        round_fair_rows.append({
            "Year": yr, "AUC": round(auc_yr,4),
            "|DPD| unfair": round(dpd_yr_raw,4),
            "|EOD| unfair": round(eod_yr_raw,4),
            "|DPD| fair":   round(dpd_yr_fair,4),
            "|EOD| fair":   round(eod_yr_fair,4),
        })

    fair_panel_df = pd.DataFrame(round_fair_rows)
    fair_panel_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_by_round.csv"), index=False)


hdr("PART 12: Statistical Significance Tests")

comparisons = [
    ("XGB GA vs LR",       p_ga,  p_lr),
    ("XGB GA vs XGB Def",  p_ga,  p_def),
    ("XGB GA vs RF",       p_ga,  p_rf),
    ("XGB GA vs LightGBM", p_ga,  p_lgb),
]
stat_rows = []
print(f"\n  {'Comparison':<30} {'Δ AUC':>8}  {'p-value':>8}  Sig")
for name, p1, p2 in comparisons:
    delta, pval = bootstrap_auc_diff_pval(y_te22, p1, p2, w_te22, n_boot=1000)
    sig = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else "ns"
    print(f"  {name:<30} {delta:>+8.4f}  {pval:>8.4f}  {sig}")
    stat_rows.append({"Comparison": name, "Delta AUC": round(delta,4),
                       "p-value": round(pval,4), "Significance": sig})

if len(dist_stats) > 0:
    stat_rows.append({"Comparison": "Moran's I", "Delta AUC": round(r_moran,4),
                       "p-value": round(p_moran,4),
                       "Significance": "***" if p_moran<0.001 else "*"})
    stat_rows.append({"Comparison": "District obs vs pred (r)", "Delta AUC": round(r_dist,4),
                       "p-value": round(p_dist,4), "Significance": "***"})

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)


hdr("PART 13: Generating Figures")

national_prev22 = np.average(df22["stunted"], weights=df22["survey_weight"])*100

fig, ax = plt.subplots(figsize=(7.2, 3.8))
gens = range(len(gen_max_vals))
ax.fill_between(gens, gen_mean_vals, gen_max_vals, alpha=0.12, color=BLUE)
ax.plot(gens, gen_max_vals,  color=BLUE, lw=2.0, marker="o", ms=3,
        label=f"Best AUC (final={BEST_CV_AUC:.4f})")
ax.plot(gens, gen_mean_vals, color=TEAL, lw=1.4, ls="--", label="Mean pop. AUC")
ax.axhline(boot_results["XGBoost (GA-optimised)"]["auc"][0],
           color=CORAL, lw=0.8, ls=":", alpha=0.6,
           label=f"Test AUC ({boot_results['XGBoost (GA-optimised)']['auc'][0]:.4f})")
ax.set_xlabel("Generation"); ax.set_ylabel("Survey-Weighted AUC (5-fold CV)")
ax.set_title(f"Genetic Algorithm Convergence — P={GA_POP}, G={GA_NGEN}\n"
             f"BDHS 2022 Development Cohort | n={len(df22):,}", fontsize=9.5)
ax.legend(fontsize=8, framealpha=0.95)
ax.grid(lw=0.3, color=GREY70, alpha=0.4)
plt.tight_layout(); savefig("fig01_ga_convergence.png")

fig, ax = plt.subplots(figsize=(8.0, 4.2))
mnames = list(boot_results.keys())
ypos   = list(range(len(mnames)-1, -1, -1))
mcols  = [GREY40, GREEN, GREY40, BLUE, TEAL]
for i, (y, nm) in enumerate(zip(ypos, mnames)):
    pt, lo, hi = boot_results[nm]["auc"]
    col = mcols[i]
    ax.plot([lo,hi],[y,y], color=col, lw=1.8, solid_capstyle="round")
    mk  = "D" if "GA" in nm else "o"
    ax.plot(pt, y, marker=mk, color=col, ms=8 if mk=="D" else 7,
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.text(hi+0.003, y, f"{pt:.3f} [{lo:.3f}–{hi:.3f}]", va="center", fontsize=7.5)
ax.set_yticks(ypos); ax.set_yticklabels(mnames, fontsize=9)
ax.set_xlabel("Survey-Weighted AUC (95% PSU-Cluster Bootstrap CI)")
ax.set_title(f"Model Performance — BDHS 2022 | {N_BOOT}-rep PSU Bootstrap", fontsize=9.5)
ax.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)
plt.tight_layout(); savefig("fig02_auc_forest_plot.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plt.sca(axes[0])
shap.summary_plot(shap_te, X_te22, feature_names=FEATS22,
                  show=False, max_display=20, plot_size=None)
axes[0].set_title("SHAP Beeswarm (top-20)\nXGBoost GA-opt | BDHS 2022", fontsize=10)

top20_rev = feat_imp.head(20).iloc[::-1]
axes[1].barh(range(len(top20_rev)), top20_rev.values,
             color=[DOMAIN_COLORS.get(feat_domain(f), GREY40) for f in top20_rev.index],
             edgecolor="white", height=0.72, linewidth=0.4)
axes[1].set_yticks(range(len(top20_rev)))
axes[1].set_yticklabels(top20_rev.index, fontsize=8, family="monospace")
axes[1].set_xlabel("Mean |SHAP value|")
axes[1].set_title("Feature Importance by Domain", fontsize=10)
legend_patches = [mpatches.Patch(color=v, label=k)
                  for k,v in DOMAIN_COLORS.items()
                  if any(feat_domain(f)==k for f in top20_rev.index)]
axes[1].legend(handles=legend_patches, fontsize=7.5, loc="lower right")
axes[1].grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)
plt.tight_layout(); savefig("fig03_shap_importance.png")

fig, ax = plt.subplots(figsize=(8.0, 5.0))
fold_cols = [f"Fold{i+1}" for i in range(5)]
stab_plot = shap_stab.sort_values("mean", ascending=True)
for i, feat in enumerate(stab_plot.index):
    fold_vals = stab_plot.loc[feat, fold_cols].values.astype(float)
    ax.plot([fold_vals.min(), fold_vals.max()], [i,i],
            color=BLUE, lw=2.5, alpha=0.35, solid_capstyle="round")
    ax.plot(stab_plot.loc[feat,"mean"], i, "o", color=BLUE, ms=7, zorder=5,
            markeredgecolor="white", markeredgewidth=0.5)
    for fv in fold_vals:
        ax.plot(fv, i, "|", color=TEAL, ms=9, markeredgewidth=1.3, alpha=0.75)
    ax.text(stab_plot.loc[feat,"mean"]*1.3, i,
            f"CV={stab_plot.loc[feat,'cv']:.1f}%", va="center", fontsize=7, color=GREY40)
ax.set_yticks(range(len(stab_plot)))
ax.set_yticklabels(stab_plot.index, fontsize=8.5, family="monospace")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP Cross-Fold Stability — Top-10 Predictors\n"
             "Ticks = individual folds | Dot = mean | CV = coefficient of variation", fontsize=9.5)
legend_elems = [
    Line2D([0],[0], color=BLUE, lw=2.5, alpha=0.35, label="Range (min–max)"),
    Line2D([0],[0], marker="o", color=BLUE, ms=6, label="Cross-fold mean",
           markeredgecolor="white", ls="None"),
    Line2D([0],[0], marker="|", color=TEAL, ms=9, label="Individual fold",
           ls="None", markeredgewidth=1.3),
]
ax.legend(handles=legend_elems, fontsize=8, framealpha=0.95, loc="lower right")
ax.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)
plt.tight_layout(); savefig("fig04_shap_stability.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
WLBL = ["Poorest","Poorer","Middle","Richer","Richest"]
cols4 = [GREY40, AMBER, CORAL, BLUE]

ax = axes[0]
for mi, (nm, fnrs) in enumerate(fnr_by_quintile.items()):
    ls_styles = ["--", "-.", ":", "-"]
    ax.plot(WLBL, fnrs, marker=["s","^","v","o"][mi], color=cols4[mi],
            lw=1.6 if mi==3 else 1.2, ms=7, label=nm, ls=ls_styles[mi])
ax.set_ylabel("False Negative Rate (%) — Missed Stunted Children")
ax.set_title("Detection Miss Rate by Wealth Quintile", fontsize=9)
ax.legend(fontsize=7, framealpha=0.95); ax.grid(lw=0.3, color=GREY70, alpha=0.4)

ax2 = axes[1]
xf   = np.arange(len(fair_models))
dpds = [abs(fair_table[n]["dpd"]) for n in fair_models]
eods = [abs(fair_table[n]["eod"]) for n in fair_models]
ax2.bar(xf-0.18, dpds, 0.35, label="|DPD|", color=BLUE, alpha=0.85, edgecolor="white")
ax2.bar(xf+0.18, eods, 0.35, label="|EOD|", color=CORAL, alpha=0.85, edgecolor="white")
for i, (d,e) in enumerate(zip(dpds,eods)):
    ax2.text(i-0.18, d+0.005, f"{d:.3f}", ha="center", fontsize=7)
    ax2.text(i+0.18, e+0.005, f"{e:.3f}", ha="center", fontsize=7)
ax2.set_xticks(xf)
ax2.set_xticklabels(list(fair_models.keys()), fontsize=7.5, rotation=15, ha="right")
ax2.set_ylabel("Fairness metric (0 = ideal)")
ax2.set_title("DPD & EOD by Model", fontsize=9)
ax2.legend(fontsize=8); ax2.grid(axis="y", lw=0.3, color=GREY70, alpha=0.4)

ax3 = axes[2]
for j, (grp, vals) in enumerate(extended_fair.items()):
    ax3.barh(grp, vals["DPD"], 0.35, color=BLUE, alpha=0.8,
             label="|DPD|" if j==0 else "")
    ax3.barh(grp, vals["EOD"], 0.35, left=vals["DPD"], color=CORAL, alpha=0.6,
             label="|EOD|" if j==0 else "")
ax3.set_xlabel("Fairness metric value")
ax3.set_title("Extended Fairness Subgroups", fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)
fig.suptitle("Fairness Analysis — BDHS 2022 | Wealth-stratified & Extended Subgroups",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout(); savefig("fig05_fairness.png")

if RUN_ABLATION:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    y_abl  = list(range(len(abl_df)-1, -1, -1))
    highlight = [BLUE if r["Configuration"]=="Full SAFE-XAI"
                 else AMBER if r["Configuration"]=="Default XGB + Fairness"
                 else CORAL for _, r in abl_df.iterrows()]

    ax = axes[0]
    for i, (y, row) in enumerate(zip(y_abl, abl_df.itertuples())):
        ax.plot([row.AUC_lo, row.AUC_hi],[y,y], color=highlight[i], lw=1.8)
        ax.plot(row.AUC, y, "D" if i==0 else ("s" if i==len(abl_df)-1 else "o"),
                color=highlight[i], ms=8 if i==0 else 6,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax.text(row.AUC_hi+0.002, y, f"{row.AUC:.4f}", va="center", fontsize=7.5)
    ax.axvline(abl_df["AUC"].iloc[0], color=BLUE, lw=0.7, ls="--", alpha=0.4)
    ax.set_yticks(list(y_abl))
    ax.set_yticklabels(abl_df["Configuration"], fontsize=8.5)
    ax.set_xlabel("Survey-Weighted AUC (95% CI, 200-rep bootstrap)")
    ax.set_title("Ablation Study — AUC", fontsize=9.5)
    ax.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)

    ax2 = axes[1]
    for i, (y, row) in enumerate(zip(y_abl, abl_df.itertuples())):
        ax2.barh(y, row.EOD, height=0.6, color=highlight[i], edgecolor="white", alpha=0.85)
        ax2.text(row.EOD+0.002, y, f"{row.EOD:.4f}", va="center", fontsize=7.5)
    ax2.set_yticks(list(y_abl))
    ax2.set_yticklabels(abl_df["Configuration"], fontsize=8.5)
    ax2.set_xlabel("Equalized Odds Difference (lower = fairer)")
    ax2.set_title("Ablation Study — Fairness (EOD)", fontsize=9.5)
    ax2.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)

    patches_abl = [mpatches.Patch(color=BLUE, label="Full SAFE-XAI"),
                   mpatches.Patch(color=AMBER, label="Default XGB + Fairness"),
                   mpatches.Patch(color=CORAL, label="Ablated configuration")]
    for axi in axes:
        axi.legend(handles=patches_abl, fontsize=7.5, framealpha=0.95)
    plt.tight_layout(); savefig("fig06_ablation.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
cal_models_plot = [("LR",p_lr,GREY40,"--"),("RF",p_rf,GREEN,"-."),
                    ("XGB GA-opt.",p_ga,BLUE,"-"),("LightGBM",p_lgb,TEAL,":")]
for nm, yp, col, ls in cal_models_plot:
    pt_,pp_ = calibration_curve(y_te22, yp, n_bins=8, strategy="quantile")
    bs_     = brier_score_loss(y_te22, yp, sample_weight=w_te22)
    ax.plot(pp_, pt_, marker="o", ms=5, color=col, ls=ls, lw=1.4,
            label=f"{nm} (Brier={bs_:.4f})")
ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5,label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction Positive (Observed)")
ax.set_title("Calibration Curves", fontsize=9.5)
ax.legend(fontsize=7.5); ax.grid(lw=0.3, color=GREY70, alpha=0.4)

ax2 = axes[1]
thresholds = np.linspace(0.05, 0.60, 100)
dca_pls = [("XGB GA-opt.",p_ga,BLUE,"-",2.0),("LR",p_lr,GREY40,"--",1.2),
            ("RF",p_rf,GREEN,"-.",1.2),("LightGBM",p_lgb,TEAL,":",1.2)]
for nm, yp, col, ls, lw_ in dca_pls:
    nb_ = [net_benefit(y_te22, yp, t) for t in thresholds]
    ax2.plot(thresholds, nb_, color=col, ls=ls, lw=lw_, label=nm)
nb_all  = [net_benefit(y_te22, np.ones_like(p_ga), t) for t in thresholds]
ax2.plot(thresholds, nb_all,  color=AMBER, lw=1.0, ls=":", label="Treat all")
ax2.plot(thresholds, [0]*len(thresholds), color=GREY10, lw=0.7, ls="-", alpha=0.4, label="Treat none")
ax2.set_xlabel("Threshold Probability"); ax2.set_ylabel("Net Benefit")
ax2.set_title("Decision-Curve Analysis", fontsize=9.5)
ax2.legend(fontsize=8); ax2.grid(lw=0.3, color=GREY70, alpha=0.4)
ax2.set_xlim(0.04, 0.61)
plt.tight_layout(); savefig("fig07_calibration_dca.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
cohort_aucs = [boot_results["XGBoost (GA-optimised)"]["auc"][0], auc17]
cohort_los  = [boot_results["XGBoost (GA-optimised)"]["auc"][1], auc17_lo]
cohort_his  = [boot_results["XGBoost (GA-optimised)"]["auc"][2], auc17_hi]
cohort_lbs  = ["BDHS 2022\n(Development)", "BDHS 2017\n(Validation)"]
cols_ext    = [BLUE, TEAL]
ax = axes[0]
for i in range(2):
    ax.barh(i, cohort_aucs[i], color=cols_ext[i], alpha=0.75, height=0.5, edgecolor="white")
    ax.plot([cohort_los[i],cohort_his[i]],[i,i], color=cols_ext[i], lw=2.5)
    ax.plot(cohort_aucs[i], i, "D", color=cols_ext[i], ms=9,
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.text(cohort_his[i]+0.002, i,
            f"{cohort_aucs[i]:.4f} [{cohort_los[i]:.4f}–{cohort_his[i]:.4f}]",
            va="center", fontsize=8)
ax.set_yticks([0,1]); ax.set_yticklabels(cohort_lbs, fontsize=9)
ax.set_xlabel("Survey-Weighted AUC (95% CI)")
ax.set_title("AUC — Dev vs Validation", fontsize=9.5)
ax.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)

ax2 = axes[1]
pt17_, pp17_ = calibration_curve(y17, p17, n_bins=8, strategy="quantile")
pt22_, pp22_ = calibration_curve(y_te22, p_ga, n_bins=8, strategy="quantile")
ax2.plot(pp17_, pt17_, "o-", color=TEAL, lw=1.6, ms=6,
         label=f"2017 (Brier={brier17:.4f})")
ax2.plot(pp22_, pt22_, "s--", color=BLUE, lw=1.4, ms=5,
         label=f"2022 (Brier={brier_score_loss(y_te22,p_ga,sample_weight=w_te22):.4f})")
ax2.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4,label="Perfect")
ax2.set_xlabel("Predicted Probability"); ax2.set_ylabel("Observed Fraction")
ax2.set_title(f"Calibration Curves\nSlope={cal_slope17:.3f}  Intercept={cal_intercept17:.3f}", fontsize=9.5)
ax2.legend(fontsize=8); ax2.grid(lw=0.3, color=GREY70, alpha=0.4)

ax3 = axes[2]
fpr22, tpr22, _ = roc_curve(y_te22, p_ga)
fpr17, tpr17, _ = roc_curve(y17, p17)
ax3.plot(fpr22, tpr22, color=BLUE, lw=2.0,
         label=f"2022 (AUC={boot_results['XGBoost (GA-optimised)']['auc'][0]:.4f})")
ax3.plot(fpr17, tpr17, color=TEAL, lw=1.6, ls="--",
         label=f"2017 (AUC={auc17:.4f})")
ax3.plot([0,1],[0,1],"k--",lw=0.7,alpha=0.4)
ax3.set_xlabel("1-Specificity"); ax3.set_ylabel("Sensitivity")
ax3.set_title("ROC — Dev vs Validation", fontsize=9.5)
ax3.legend(fontsize=8, loc="lower right"); ax3.grid(lw=0.3, color=GREY70, alpha=0.4)
fig.suptitle("External Temporal Validation — Frozen 2022 GA-XGBoost → BDHS 2017",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout(); savefig("fig08_external_validation.png")

if RUN_PANEL_ANALYSIS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0,0]
    for outcome, col in [("Stunting",BLUE),("Underweight",CORAL),("Wasting",AMBER)]:
        sub_t = traj_df[traj_df["Outcome"]==outcome].sort_values("Year")
        ax.plot(sub_t["Year"], sub_t["Prevalence"], "o-", color=col, lw=2, ms=8, label=outcome)
        for _, row in sub_t.iterrows():
            ax.annotate(f"{row['Prevalence']:.1f}%", (row["Year"], row["Prevalence"]),
                        textcoords="offset points", xytext=(0,8), fontsize=8, ha="center")
    ax.set_xlabel("Survey year"); ax.set_ylabel("Prevalence (%)")
    ax.set_title("Child Malnutrition Prevalence Trajectory\nBDHS 2011–2022", fontsize=10)
    ax.legend(fontsize=9); ax.grid(lw=0.3, color=GREY70, alpha=0.4)

    ax2 = axes[0,1]
    for i, row in roll_df.iterrows():
        label = f"{row['Training']}→{row['Validation']}"
        ax2.bar(i, row["AUC"], color=ROUND_COLORS.get(int(row["Validation"]), BLUE),
                width=0.65, edgecolor="white", alpha=0.85)
        ax2.text(i, row["AUC"]+0.003, f"{row['AUC']:.4f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(roll_df)))
    ax2.set_xticklabels(
        [f"{r['Training']}→{r['Validation']}" for _, r in roll_df.iterrows()],
        rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Validation AUC")
    ax2.set_title("Rolling Temporal Validation AUC", fontsize=10)
    ax2.set_ylim(0.55, 0.85)
    ax2.grid(axis="y", lw=0.3, color=GREY70, alpha=0.4)
    ax2.axhline(0.7, color=GREY40, lw=0.8, ls="--", alpha=0.5, label="AUC=0.70 reference")
    ax2.legend(fontsize=8)

    ax3 = axes[1,0]
    for yr in sorted(round_shap_vals.keys()):
        rank  = list(round_shap_vals[yr].index).index("mat_height_cm") + 1 \
                if "mat_height_cm" in round_shap_vals[yr].index else np.nan
        ax3.bar(str(yr), round_shap_vals[yr].get("mat_height_cm", 0),
                color=ROUND_COLORS[yr], width=0.6, edgecolor="white", alpha=0.85,
                label=f"{yr} (rank={rank:.0f})")
        ax3.text(str(yr), round_shap_vals[yr].get("mat_height_cm",0)+0.002,
                 f"Rank {rank:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Survey year"); ax3.set_ylabel("Mean |SHAP value|")
    ax3.set_title("Maternal Height SHAP Dominance Across Rounds", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(axis="y", lw=0.3, color=GREY70, alpha=0.4)

    ax4 = axes[1,1]
    drift_feats_plot = ["mat_height_cm","wealth_idx","anc_visits","WASH_score","improved_sanit"]
    for fi, f in enumerate(drift_feats_plot):
        row_d = drift_df[drift_df["Feature"]==f]
        if len(row_d) == 0: continue
        yr_cols = [str(yr) for yr in sorted(round_dfs.keys()) if str(yr) in row_d.columns]
        vals_d  = [row_d[yc].values[0] for yc in yr_cols]
        if vals_d[0] != 0:
            vals_norm = [v/vals_d[0]*100 for v in vals_d]
        else:
            vals_norm = vals_d
        ax4.plot([int(y) for y in yr_cols], vals_norm, "o-",
                 color=PAL6[fi], lw=1.6, ms=6, label=f)
    ax4.axhline(100, color=GREY40, lw=0.8, ls="--", alpha=0.5, label="2011 baseline")
    ax4.set_xlabel("Survey year")
    ax4.set_ylabel("Value as % of 2011 baseline")
    ax4.set_title("Covariate Drift 2011→2022", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(lw=0.3, color=GREY70, alpha=0.4)

    fig.suptitle("Panel Analysis — BDHS 2011–2022",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(); savefig("fig09_panel_analysis.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(fair_panel_df["Year"], fair_panel_df["|DPD| unfair"], "o-",
            color=CORAL, lw=2, ms=8, label="|DPD| (unfair model)")
    ax.plot(fair_panel_df["Year"], fair_panel_df["|DPD| fair"], "s--",
            color=BLUE, lw=2, ms=8, label="|DPD| (fairness-calibrated)")
    ax.fill_between(fair_panel_df["Year"],
                     fair_panel_df["|DPD| unfair"], fair_panel_df["|DPD| fair"],
                     alpha=0.1, color=CORAL)
    ax.set_xlabel("Survey year"); ax.set_ylabel("|DPD| (0 = ideal)")
    ax.set_title("Demographic Parity Gap Over Time", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(lw=0.3, color=GREY70, alpha=0.4)

    ax2 = axes[1]
    ax2.plot(fair_panel_df["Year"], fair_panel_df["|EOD| unfair"], "o-",
             color=CORAL, lw=2, ms=8, label="|EOD| (unfair model)")
    ax2.plot(fair_panel_df["Year"], fair_panel_df["|EOD| fair"], "s--",
             color=BLUE, lw=2, ms=8, label="|EOD| (fairness-calibrated)")
    ax2.set_xlabel("Survey year"); ax2.set_ylabel("|EOD| (0 = ideal)")
    ax2.set_title("Equalized Odds Gap Over Time", fontsize=9.5)
    ax2.legend(fontsize=8); ax2.grid(lw=0.3, color=GREY70, alpha=0.4)
    fig.suptitle("Per-Round Fairness Analysis — Wealth Detection Gap 2011→2022",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout(); savefig("fig10_panel_fairness.png")

df_vm = df22.dropna(subset=["mat_height_cm","stunted"]).copy()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

df_vm["height_decile"] = pd.qcut(df_vm["mat_height_cm"], 10, labels=False)
ht_grp = df_vm.groupby("height_decile").apply(lambda g: pd.Series({
    "stunting": np.average(g["stunted"], weights=g["survey_weight"])*100,
    "mean_ht":  g["mat_height_cm"].mean()})).reset_index()
axes[0].plot(ht_grp["mean_ht"], ht_grp["stunting"], "o-", color=BLUE, lw=2, ms=7)
axes[0].fill_between(ht_grp["mean_ht"],
                      ht_grp["stunting"]*0.88, ht_grp["stunting"]*1.12,
                      alpha=0.1, color=BLUE)
axes[0].axvline(145, color=CORAL, ls="--", lw=1.5, label="Short stature (<145cm)")
axes[0].axhline(national_prev22, color=GREY40, ls=":", lw=1, alpha=0.7,
                label=f"National ({national_prev22:.1f}%)")
axes[0].set_xlabel("Maternal height (cm)")
axes[0].set_ylabel("Stunting prevalence (%)")
axes[0].set_title("Maternal Height vs\nChild Stunting (by decile)")
axes[0].legend(fontsize=8); axes[0].grid(lw=0.3, color=GREY70, alpha=0.4)

bmi_cats = {
    "Underweight\n(<18.5)":  (df_vm["mat_bmi"]<18.5),
    "Normal\n(18.5-25)":     df_vm["mat_bmi"].between(18.5, 24.9),
    "Overweight\n(25-30)":   df_vm["mat_bmi"].between(25, 29.9),
    "Obese\n(≥30)":          (df_vm["mat_bmi"]>=30),
}
cat_p, cat_l = [], []
for lbl, mask in bmi_cats.items():
    g = df_vm[mask]
    if len(g) > 10:
        cat_p.append(np.average(g["stunted"], weights=g["survey_weight"])*100)
        cat_l.append(lbl)
bars_ = axes[1].bar(cat_l, cat_p,
                     color=[CORAL,BLUE,AMBER,PURPLE][:len(cat_p)],
                     alpha=0.85, width=0.55, edgecolor="white")
for bar, val in zip(bars_, cat_p):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
axes[1].set_ylabel("Stunting prevalence (%)")
axes[1].set_title("Maternal BMI Category\nvs Child Stunting")
axes[1].grid(axis="y", lw=0.3, color=GREY70, alpha=0.4)

if "mat_height_cm" in FEATS22 and "mat_bmi" in FEATS22:
    idx_ht  = FEATS22.index("mat_height_cm")
    idx_bmi = FEATS22.index("mat_bmi")
    sc = axes[2].scatter(X_te22[:,idx_ht], shap_te[:,idx_ht],
                          c=X_te22[:,idx_bmi], cmap="RdYlGn_r",
                          s=12, alpha=0.45)
    plt.colorbar(sc, ax=axes[2], label="Maternal BMI")
    axes[2].axhline(0, color=GREY40, lw=0.8, ls="--")
    axes[2].axvline(145, color=CORAL, lw=1.5, ls="--", label="<145cm")
    axes[2].set_xlabel("Maternal height (cm)")
    axes[2].set_ylabel("SHAP value")
    axes[2].set_title("SHAP Dependence: Maternal Height\n(coloured by maternal BMI)")
    axes[2].legend(fontsize=8); axes[2].grid(lw=0.3, color=GREY70, alpha=0.4)
fig.suptitle("Intergenerational Nutritional Risk — Maternal Anthropometry (BDHS 2022)",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout(); savefig("fig11_maternal_anthropometry.png")

if len(dist_stats) > 0:
    cmap_risk = LinearSegmentedColormap.from_list("risk", [GREEN,AMBER,CORAL,"#7b0000"])
    norm_r    = plt.Normalize(dist_stats["predicted_stunting"].min(),
                               dist_stats["predicted_stunting"].max())
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.48)

    ax1   = fig.add_subplot(gs[0,:2])
    top20d = dist_stats.nlargest(20,"predicted_stunting").reset_index(drop=True)
    ax1.barh(range(20), top20d["predicted_stunting"],
             color=[cmap_risk(norm_r(v)) for v in top20d["predicted_stunting"]],
             height=0.75, edgecolor="white", linewidth=0.3)
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([f"District {int(d)}" for d in top20d["district"]], fontsize=8.5)
    for i, (val, obs) in enumerate(zip(top20d["predicted_stunting"],
                                        top20d["observed_stunting"])):
        ax1.text(val+0.3, i, f"{val:.1f}% (obs:{obs:.1f}%)", va="center", fontsize=7.5)
        ax1.plot(obs, i, "|", color=TEAL, ms=9, markeredgewidth=1.5, zorder=5)
    ax1.axvline(national_prev22, color=GREY10, ls="--", lw=1.5,
                label=f"National ({national_prev22:.1f}%)")
    ax1.set_xlabel("LOO-CV Predicted Stunting (%)")
    ax1.set_title("Top 20 Highest-Risk Districts — 2022", fontsize=9.5)
    ax1.legend(fontsize=8); ax1.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)

    ax2 = fig.add_subplot(gs[0,2])
    tc = dist_stats["risk_tier"].value_counts()
    ax2.pie(tc.values, labels=tc.index, autopct="%1.0f%%",
            colors=[GREEN,AMBER,CORAL,"#7b0000"][:len(tc)],
            startangle=90, textprops={"fontsize":8.5},
            wedgeprops={"edgecolor":"white","linewidth":1.2})
    ax2.set_title(f"District Risk Tiers\n({len(dist_stats)} districts)", fontsize=9.5)

    ax3 = fig.add_subplot(gs[1,0])
    sc  = ax3.scatter(dist_stats["observed_stunting"], dist_stats["predicted_stunting"],
                       c=dist_stats["mean_wealth"], cmap="RdYlGn",
                       s=dist_stats["n_children"]*0.5+15, alpha=0.75,
                       edgecolors="white", lw=0.5)
    ax3.plot([0,55],[0,55],"k--",lw=0.8,alpha=0.4)
    plt.colorbar(sc, ax=ax3, label="Mean wealth index")
    ax3.set_xlabel("Observed stunting (%)")
    ax3.set_ylabel("Predicted stunting (%)")
    ax3.set_title(f"Obs vs Pred by District\n(r={r_dist:.3f}, p<0.001)", fontsize=9.5)
    ax3.grid(lw=0.3, color=GREY70, alpha=0.4)

    div_stats = df22.groupby("division").apply(lambda g: pd.Series({
        "stunting":   np.average(g["stunted"], weights=g["survey_weight"])*100,
        "wasted":     np.average(g["wasted"].fillna(0), weights=g["survey_weight"])*100,
    })).reset_index()
    div_stats["name"] = div_stats["division"].map(DIV_LABELS)
    ax4 = fig.add_subplot(gs[1,1])
    div_s = div_stats.sort_values("stunting", ascending=True)
    yd_   = np.arange(len(div_s))
    ax4.barh(yd_, div_s["stunting"], height=0.55, color=BLUE, alpha=0.85, label="Stunting")
    ax4.barh(yd_, div_s["wasted"], height=0.55, color=CORAL, alpha=0.5, label="Wasting")
    ax4.set_yticks(yd_); ax4.set_yticklabels(div_s["name"], fontsize=9)
    ax4.set_xlabel("Prevalence (%)")
    ax4.set_title("Stunting & Wasting by Division", fontsize=9.5)
    ax4.legend(fontsize=8); ax4.grid(axis="x", lw=0.3, color=GREY70, alpha=0.4)

    ax5 = fig.add_subplot(gs[1,2])
    wq_data = []
    for q in range(1, 6):
        g = df22[df22["V190"]==q]
        if len(g) > 5:
            wq_data.append((q,
                np.average(g["stunted"], weights=g["survey_weight"])*100,
                np.average(g["WASH_score"], weights=g["survey_weight"])))
    wq_df = pd.DataFrame(wq_data, columns=["q","stunting","wash"])
    sc5   = ax5.scatter(wq_df["wash"], wq_df["stunting"], c=wq_df["q"],
                         cmap="RdYlGn", s=200, zorder=5, edgecolors="black", lw=0.8)
    for _, row in wq_df.iterrows():
        ax5.annotate({1:"Poorest",2:"Poorer",3:"Middle",4:"Richer",5:"Richest"}[int(row.q)],
                     (row.wash, row.stunting), fontsize=9,
                     xytext=(4,4), textcoords="offset points")
    ax5.set_xlabel("Mean WASH Score (0-5)")
    ax5.set_ylabel("Stunting Prevalence (%)")
    ax5.set_title("WASH vs Stunting\nby Wealth Quintile", fontsize=9.5)
    ax5.grid(lw=0.3, color=GREY70, alpha=0.4)

    fig.suptitle(f"District & Division Nutritional Vulnerability Atlas — BDHS 2022",
                  fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(); savefig("fig12_district_atlas.png")


hdr("PART 14: Publication Tables")

t1_rows = []
for nm, br in boot_results.items():
    t1_rows.append({
        "Model":                nm,
        "AUC (95% CI)":         f"{br['auc'][0]:.3f} [{br['auc'][1]:.3f}–{br['auc'][2]:.3f}]",
        "Sensitivity (95% CI)": f"{br['sensitivity'][0]:.3f} [{br['sensitivity'][1]:.3f}–{br['sensitivity'][2]:.3f}]",
        "Specificity (95% CI)": f"{br['specificity'][0]:.3f} [{br['specificity'][1]:.3f}–{br['specificity'][2]:.3f}]",
        "PPV (95% CI)":         f"{br['ppv'][0]:.3f} [{br['ppv'][1]:.3f}–{br['ppv'][2]:.3f}]",
        "Brier (95% CI)":       f"{br['brier'][0]:.4f} [{br['brier'][1]:.4f}–{br['brier'][2]:.4f}]",
        "Primary": "Yes†" if nm=="XGBoost (GA-optimised)" else "",
    })
table1 = pd.DataFrame(t1_rows)

table2 = pd.DataFrame([
    {"Model": nm,
     "|DPD| (ideal=0)": round(abs(fair_table[nm]["dpd"]), 4),
     "|EOD| (ideal=0)": round(abs(fair_table[nm]["eod"]), 4),
     "AUC":             round(fair_table[nm]["auc"], 4)}
    for nm in fair_table
])

table3 = pd.DataFrame([
    {"Rank": i+1, "Feature": f, "Mean |SHAP|": round(v, 4),
     "Domain": feat_domain(f),
     "CV (%)": round(shap_stab.loc[f,"cv"], 1) if f in shap_stab.index else "—"}
    for i, (f,v) in enumerate(feat_imp.head(20).items())
])

table4 = abl_df if RUN_ABLATION else pd.DataFrame()

ext_val_dict = {
    "Cohort":               ["BDHS 2022 (Development)", "BDHS 2017 (Validation)"],
    "n":                    [len(df22), len(df17)],
    "Prevalence (%)":       [round(national_prev22,1), round(prev17,1)],
    "AUC":                  [boot_results["XGBoost (GA-optimised)"]["auc"][0], round(auc17,4)],
    "AUC 95% CI lo":        [boot_results["XGBoost (GA-optimised)"]["auc"][1], round(auc17_lo,4)],
    "AUC 95% CI hi":        [boot_results["XGBoost (GA-optimised)"]["auc"][2], round(auc17_hi,4)],
    "Brier":                [boot_results["XGBoost (GA-optimised)"]["brier"][0], round(brier17,4)],
    "Calibration Slope":    ["—", round(cal_slope17, 4)],
    "Calibration Intercept":["—", round(cal_intercept17, 4)],
    "Brier (recalibrated)": ["—", round(brier17_recal, 4)],
    "|DPD|":                [round(abs(fair_table["XGB GA"]["dpd"]),4), round(dpd17,4)],
    "|EOD|":                [round(abs(fair_table["XGB GA"]["eod"]),4), round(eod17,4)],
}
table5 = pd.DataFrame(ext_val_dict)

if RUN_MICE and len(imputed_datasets)>1:
    table6_mice = pd.DataFrame({
        "Imputation": list(range(1, N_MICE_IMPUTATIONS+1)),
        "AUC": [round(a,4) for a in mice_aucs],
    })
    table6_mice.loc[len(table6_mice)] = ["Rubin pooled", round(pooled_auc,4)]
    table6_mice.loc[len(table6_mice)] = ["SE", round(pooled_se,4)]
else:
    table6_mice = pd.DataFrame()

table7  = harm_report
table8  = roll_df if RUN_PANEL_ANALYSIS else pd.DataFrame()
table9  = shap_cross if RUN_PANEL_ANALYSIS else pd.DataFrame()
table10 = traj_df.pivot(index="Outcome", columns="Year", values="Prevalence").reset_index() \
           if RUN_PANEL_ANALYSIS else pd.DataFrame()
table11 = fair_panel_df if RUN_PANEL_ANALYSIS else pd.DataFrame()
table12 = drift_df if RUN_PANEL_ANALYSIS else pd.DataFrame()
table13 = stat_df
table14 = miss_table
table15 = multi_df
table16 = sens_df
table17 = sg_df

xlsx_path = os.path.join(OUTPUT_DIR, "publication_tables.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    table1.to_excel(writer,  sheet_name="T1_Performance",       index=False)
    table2.to_excel(writer,  sheet_name="T2_Fairness",          index=False)
    table3.to_excel(writer,  sheet_name="T3_SHAP_Top20",        index=False)
    if len(table4): table4.to_excel(writer, sheet_name="T4_Ablation",     index=False)
    table5.to_excel(writer,  sheet_name="T5_ExtVal",             index=False)
    if len(table6_mice): table6_mice.to_excel(writer, sheet_name="T6_MICE", index=False)
    table7.to_excel(writer,  sheet_name="T7_Harmonization",     index=False)
    if len(table8): table8.to_excel(writer, sheet_name="T8_RollingVal",    index=False)
    if len(table9): table9.to_excel(writer, sheet_name="T9_SHAP_CrossRound", index=True)
    if len(table10): table10.to_excel(writer, sheet_name="T10_Trajectory", index=False)
    if len(table11): table11.to_excel(writer, sheet_name="T11_RoundFairness", index=False)
    if len(table12): table12.to_excel(writer, sheet_name="T12_CovDrift",   index=False)
    table13.to_excel(writer, sheet_name="T13_StatTests",         index=False)
    table14.to_excel(writer, sheet_name="T14_MissingData",       index=False)
    table15.to_excel(writer, sheet_name="T15_MultiOutcome",      index=False)
    table16.to_excel(writer, sheet_name="T16_Sensitivity",       index=False)
    table17.to_excel(writer, sheet_name="T17_Subgroups",         index=False)
    shap_stab.to_excel(writer, sheet_name="SHAP_Stability",      index=True)
    if len(dist_stats): dist_stats.to_excel(writer, sheet_name="DistrictAtlas", index=False)

print(f"  Saved publication_tables.xlsx (17 sheets)")


hdr("FINAL RESULTS SUMMARY")
print(f"  Total runtime: {time.time()-t0:.0f}s")

print(f"\n  Cohort sizes:")
for yr in sorted(round_dfs):
    n   = len(round_dfs[yr])
    prv = np.average(round_dfs[yr]["stunted"], weights=round_dfs[yr]["survey_weight"])*100
    print(f"  BDHS {yr}: n={n:,}  stunting={prv:.1f}%")
print(f"  Panel total: n={len(df_panel):,}")

print(f"\n  Primary model (BDHS 2022):")
for nm, br in boot_results.items():
    a, lo, hi = br["auc"]
    b         = br["brier"][0]
    print(f"  {nm:<28} AUC={a:.4f} [{lo:.4f}–{hi:.4f}]  Brier={b:.4f}")

print(f"\n  External validation (BDHS 2017, frozen model):")
print(f"  AUC={auc17:.4f} [{auc17_lo:.4f}–{auc17_hi:.4f}]")
print(f"  Calibration slope={cal_slope17:.4f}  intercept={cal_intercept17:.4f}")
print(f"  Brier uncalibrated={brier17:.4f}  recalibrated={brier17_recal:.4f}")

print(f"\n  Top 5 SHAP predictors (BDHS 2022):")
for i, (f,v) in enumerate(feat_imp.head(5).items(), 1):
    cv_v = shap_stab.loc[f,"cv"] if f in shap_stab.index else 0
    print(f"  {i}. {f:<30} SHAP={v:.4f}  CV={cv_v:.1f}%")

print(f"\n  Fairness (4 models):")
for nm, vals in fair_table.items():
    print(f"  {nm:<28} |DPD|={abs(vals['dpd']):.4f}  |EOD|={abs(vals['eod']):.4f}")

if RUN_ABLATION:
    print(f"\n  Ablation (6 configs):")
    for _, row in abl_df.iterrows():
        print(f"  {row['Configuration']:<35} AUC={row['AUC']:.4f}  EOD={row['EOD']:.4f}")

if RUN_PANEL_ANALYSIS:
    print(f"\n  Panel: mat_height SHAP rank by round:")
    for yr in sorted(round_shap_ranks.keys()):
        rank = round_shap_ranks[yr].get("mat_height_cm","?")
        val  = round_shap_vals[yr].get("mat_height_cm",0)
        print(f"  {yr}: rank={rank}  |SHAP|={val:.4f}")

print(f"\n  Outputs saved to {OUTPUT_DIR}")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
    print(f"  {fn:<55} {sz/1024:>7.1f} KB")

print(f"\n{'='*72}")
print(f"  SAFE-XAI COMPLETE.")
print(f"{'='*72}")