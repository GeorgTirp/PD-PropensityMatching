import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from icecream import ic
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from pathlib import Path
import csv
import os
import re
from datetime import datetime
import copy


def _normalize_date_series(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    """
    Normalize messy date strings so pandas can parse them consistently.
    - Convert to string, strip, unify separators to '-'
    - Add day '01' for YYYY-MM and MM-YYYY cases
    - Add '01-01' for YYYY-only
    - Fix accidental YYYY-DD-MM (swap last two if middle > 12 and last <= 12)
    """
    s = s.astype(str).str.strip()
    s = s.replace({'': np.nan, 'NaT': np.nan, 'nan': np.nan})
    s = s.dropna()

    # unify separators
    s = s.str.replace(r'[./]', '-', regex=True)

    # YEAR-MONTH -> append -01
    mask_year_month = s.str.match(r'^\d{4}-\d{1,2}$')
    s.loc[mask_year_month] = s.loc[mask_year_month] + '-01'

    # MONTH-YEAR (e.g., 09-2022) -> prepend 01-
    mask_month_year = s.str.match(r'^\d{1,2}-\d{4}$')
    s.loc[mask_month_year] = '01-' + s.loc[mask_month_year]

    # YEAR only -> append -01-01
    mask_year_only = s.str.match(r'^\d{4}$')
    s.loc[mask_year_only] = s.loc[mask_year_only] + '-01-01'

    # Fix accidental YYYY-DD-MM (e.g., 2013-21-11 -> 2013-11-21)
    mask_ymd = s.str.match(r'^\d{4}-\d{1,2}-\d{1,2}$')
    def _swap_if_needed(x: str) -> str:
        y, m, d = x.split('-')
        mi, di = int(m), int(d)
        if mi > 12 and di <= 12:
            return f"{y}-{di:02d}-{mi:02d}"
        return x
    s.loc[mask_ymd] = s.loc[mask_ymd].apply(_swap_if_needed)

    return s


def safe_parse_dates(df: pd.DataFrame, cols, dayfirst: bool = True, report: bool = True) -> pd.DataFrame:
    """
    Parse mixed-format date columns safely.
    - Pre-normalizes strings
    - Uses pandas >=2.0 format='mixed' when available
    - Falls back to elementwise parsing if needed
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        s = df[col].copy()
        s_norm = s.copy()
        mask_nonnull = s_norm.notna()
        s_norm.loc[mask_nonnull] = _normalize_date_series(s_norm.loc[mask_nonnull], dayfirst=dayfirst)

        try:
            parsed = pd.to_datetime(s_norm, errors='coerce', dayfirst=dayfirst, format='mixed')
        except TypeError:
            def parse_one(x):
                if pd.isna(x): return pd.NaT
                try:
                    return pd.to_datetime(x, errors='coerce', dayfirst=dayfirst)
                except Exception:
                    return pd.NaT
            parsed = s_norm.apply(parse_one)

        df[col] = parsed

        if report:
            n_total = len(df)
            n_ok = df[col].notna().sum()
            n_bad = n_total - n_ok
            sample_bad = df.loc[df[col].isna(), col].head(3)
            print(f"[safe_parse_dates] {col}: parsed {n_ok}/{n_total} ({n_bad} NaT). "
                  f"Examples of failures:\n{sample_bad}")

    return df


class Data:
    MOCA_CATEGORY_MAP = {
        "MoCA_Executive": ["MoCA_Executive"],
        "MoCA_Benennen": ["MoCA_Benennen"],
        "MoCA_Aufmerksamkeit": [
            "MoCA_Aufmerksamkeit_Zahlenliste",
            "MoCA_Aufmerksamkeit_Buchstabenliste",
            "MoCA_Aufmerksamkeit_Abziehen"
        ],
        "MoCA_Sprache": [
            "MoCA_Sprache_Wiederholen",
            "MoCA_Sprache_Buchstaben"
        ],
        "MoCA_Abstraktion": ["MoCA_Abstraktion"],
        "MoCA_Erinnerung": ["MoCA_Erinnerung"],
        "MoCA_Orientierung": ["MoCA_Orientierung"]
    }

    MOCA_DATA_COLS = list(MOCA_CATEGORY_MAP.keys())

    def __init__(self, path_of_folder: str, foldertype: str = "PPMI", covariate_names: Optional[str] = None):
        """
        No large dataframes are stored on the instance.
        We only keep the base path, folder type, and the optional mapping file path.
        """
        self.path = path_of_folder
        if foldertype not in ("PPMI", "tuebingen", "custom"):
            raise ValueError("foldertype must be one of {'PPMI','tuebingen','custom'}")
        self.foldertype = foldertype
        self.covariate_names = covariate_names  # can be set later via load_ppmi(...)

    # ---------- Loading (returns dict) ----------
    def load_ppmi(self, covariate_names: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load PPMI-layout CSVs and return a dict of DataFrames.
        Also stores the covariate mapping path on the instance (so match_* doesn't need it).
        """
        if self.foldertype != "PPMI":
            raise ValueError("For foldertype, only 'PPMI' is allowed for load_ppmi().")

        if covariate_names is not None:
            self.covariate_names = covariate_names

        datdir = self.path

        stat = pd.read_csv(f"{datdir}/Participant_Status_04Sep2025.csv")
        dbs  = pd.read_csv(f"{datdir}/Surgery_for_PD_Log_15Feb2024.csv")
        dbs  = dbs.loc[dbs.PDSURGTP == 1, :]
        medication = pd.read_csv(f"{datdir}/LEDD_Concomitant_Medication_Log_24Apr2024.csv")
        diaghist = pd.read_csv(f"{datdir}/PD_Diagnosis_History_21Feb2024.csv")

        demo = pd.read_csv(f"{datdir}/Demographics_04Sep2025.csv")
        demo['SEX'] = demo['SEX'].replace({
            1: 'male', 1.0: 'male', '1': 'male',
            0: 'female', 0.0: 'female', '0': 'female'
        })

        updrs1  = pd.read_csv(f"{datdir}/MDS-UPDRS_Part_I_04Sep2025.csv")
        updrs1p = pd.read_csv(f"{datdir}/MDS-UPDRS_Part_I_Patient_Questionnaire_04Sep2025.csv")
        updrs1  = pd.merge(updrs1, updrs1p, on='PATNO', how='outer')

        updrs2p = pd.read_csv(f"{datdir}/MDS_UPDRS_Part_II__Patient_Questionnaire_04Sep2025.csv")

        updrs3  = pd.read_csv(f"{datdir}/MDS-UPDRS_Part_III_04Sep2025.csv")
        UPDRSrig = ["NP3RIGLL", "NP3RIGLU", "NP3RIGN", "NP3RIGRL", "NP3RIGRU"]
        updrs3['rigidity'] = updrs3.loc[:, UPDRSrig].sum(1)

        updrs3["bradykinesia"] = updrs3.loc[:, "NP3BRADY"]
        UPDRSlat = ["NP3FTAP", "NP3HMOV", "NP3KTRM", "NP3LGAG", "NP3PRSP", "NP3PTRM"]
        updrs3['latindex'] = (updrs3.loc[:, [i+"R" for i in UPDRSlat]].sum(1)
                              - updrs3.loc[:, [i+"L" for i in UPDRSlat]].sum(1))
        latix = updrs3.groupby(["PATNO", "EVENT_ID", "PDSTATE"], as_index=False).latindex.mean()
        updrs3 = pd.merge(updrs3, latix)

        updrs4 = pd.read_csv(f"{datdir}/MDS-UPDRS_Part_IV__Motor_Complications_04Sep2025.csv")
        mds_updrs = dict(mds_updrs1=updrs1, mds_updrs2=updrs2p, mds_updrs3=updrs3, mds_updrs4=updrs4)

        moca = pd.read_csv(f"{datdir}/Montreal_Cognitive_Assessment__MoCA__12Mar2024.csv")
        rbd  = pd.read_csv(f"{datdir}/REM_Sleep_Behavior_Disorder_Questionnaire_08Feb2024.csv")

        return {
            "stat": stat, "dbs": dbs, "medication": medication, "diaghist": diaghist,
            "demo": demo, "mds_updrs": mds_updrs, "moca": moca, "rbd": rbd
        }

    # ---------- Conversion (pure function style) ----------
    def convert_to_standard_keys(self, complete_data: Dict[str, pd.DataFrame], DBS: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Convert a raw PPMI dict to the standard keys based on the tuebingen format.
        Returns a **new** converted dict. Does not mutate self or the input dict.
        Uses self.covariate_names (must be set beforehand, e.g. via load_ppmi(...)).
        """
        if not isinstance(complete_data, dict):
            raise ValueError("complete_data must be a dict of DataFrames")

        if self.covariate_names is None:
            raise ValueError("covariate_names not set. Call load_ppmi(covariate_names=...) first.")

        cov_path = Path(str(self.covariate_names)).expanduser().resolve()
        if not cov_path.exists():
            raise FileNotFoundError(f"Covariate names file not found at {cov_path}")

        covariate_dict = {}
        with open(cov_path, mode='r') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            for rows in reader:
                key = rows[0]
                covariate_dict[key] = rows[1:]

        cd = copy.deepcopy(complete_data)

        # ---- UPDRS renames (positional as in your original code) ----
        cd['mds_updrs']['mds_updrs1'].columns.values[6:12] = covariate_dict['mds_updrs1'][5:11]
        cd['mds_updrs']['mds_updrs1'].columns.values[5:19] = covariate_dict['mds_updrs1'][4:18]

        cd['mds_updrs']['mds_updrs2'].columns.values[6:20] = covariate_dict['mds_updrs2'][5:19]
        cd['mds_updrs']['mds_updrs2'].rename(columns={'INFODT': covariate_dict['mds_updrs3'][4]}, inplace=True)

        cd['mds_updrs']['mds_updrs3'].columns.values[23:56] = covariate_dict['mds_updrs3'][8:41]
        cd['mds_updrs']['mds_updrs3'].rename(columns={'NHY': covariate_dict['mds_updrs3'][-2]}, inplace=True)
        cd['mds_updrs']['mds_updrs3'].rename(columns={'NP3TOT': covariate_dict['mds_updrs3'][-1]}, inplace=True)
        cd['mds_updrs']['mds_updrs3'].rename(columns={'DYSKPRES': covariate_dict['mds_updrs3'][-4]}, inplace=True)
        cd['mds_updrs']['mds_updrs3'].rename(columns={'EXAMDT': "TEST_DATUM"}, inplace=True)

        cd['mds_updrs']['mds_updrs4'].columns.values[[5, 9, 10, 14, 15, 16, 20]] = covariate_dict['mds_updrs4'][5:]
        cd['mds_updrs']['mds_updrs4'].drop(cd['mds_updrs']['mds_updrs4'].columns[[6,7,8,11,12,13,17,18,19]], axis=1, inplace=True)

        # ---- Demographics -> demo_dict ----
        demo_dict = {}
        if DBS:
            dbs_and_demo = pd.merge(cd["dbs"], cd["demo"], on="PATNO", how="inner")
            dbs_and_demo = safe_parse_dates(dbs_and_demo, cols=["BIRTHDT", "PDSURGDT"], dayfirst=True, report=True)
            demo_dict['OP_DATUM'] = dbs_and_demo['PDSURGDT']
            bdt = dbs_and_demo['BIRTHDT']
            opdt = dbs_and_demo['PDSURGDT']
            demo_dict['LOCATION'] = dbs_and_demo['PDSRGLOC']
            tp_dict = {1:"GPi",2:"STN",3:"OTHER",4:"NA",5:"UNKNOWN",6:"VIM"}
            demo_dict["LOCATION"] = demo_dict["LOCATION"].replace(tp_dict)
            demo_dict['AGE_AT_OP'] = (opdt - bdt).dt.days / 365.25
            demo_dict['SEX'] = dbs_and_demo['SEX']
            demo_dict['PATNO'] = dbs_and_demo['PATNO']

            diag_df = cd["diaghist"][["PATNO", "PDDXDT"]].copy()
            diag_df = safe_parse_dates(diag_df, cols=["PDDXDT"], dayfirst=True, report=True)
            diag_df = diag_df.dropna(subset=["PDDXDT"])
            if not diag_df.empty:
                diag_df = diag_df.sort_values("PDDXDT").drop_duplicates("PATNO", keep="first")
                diag_df = diag_df.rename(columns={"PDDXDT": "DIAG_DATE"})
            else:
                diag_df = pd.DataFrame(columns=["PATNO", "DIAG_DATE"])
            demo_df = pd.merge(pd.DataFrame(demo_dict), diag_df, on="PATNO", how="left")
        else:
            cd['demo'] = safe_parse_dates(cd['demo'], cols=["PDSURGDT"], dayfirst=True, report=True)
            #demo_dict['OP_DATUM'] = cd['demo']['PDSURGDT']
            demo_dict['SEX'] = cd['demo']['SEX']
            demo_dict['PATNO'] = cd['demo']['PATNO']
            diag_df = cd["diaghist"][["PATNO", "PDDXDT"]].copy()
            diag_df = safe_parse_dates(diag_df, cols=["PDDXDT"], dayfirst=True, report=True)
            diag_df = diag_df.dropna(subset=["PDDXDT"])
            if not diag_df.empty:
                diag_df = diag_df.sort_values("PDDXDT").drop_duplicates("PATNO", keep="first")
                diag_df = diag_df.rename(columns={"PDDXDT": "DIAG_DATE"})
            else:
                diag_df = pd.DataFrame(columns=["PATNO", "DIAG_DATE"])
            demo_df = pd.merge(pd.DataFrame(demo_dict), diag_df, on="PATNO", how="left")

        medication_df = cd.get("medication")
        if medication_df is not None and not medication_df.empty and "OP_DATUM" in demo_df.columns:
            op_series = demo_df.set_index("PATNO")["OP_DATUM"]
            ledd_series = self._compute_ledd_pre(medication_df, op_series)
            if not ledd_series.empty:
                demo_df = demo_df.merge(ledd_series, left_on="PATNO", right_index=True, how="left")
        if "LEDD_pre" not in demo_df.columns:
            demo_df["LEDD_pre"] = np.nan

        # ---- MoCA reshape ----
        moca = cd['moca']

        def _sum_columns(columns: List[str]) -> pd.Series:
            available = [col for col in columns if col in moca.columns]
            if not available:
                return pd.Series(np.nan, index=moca.index)
            numeric = moca[available].apply(pd.to_numeric, errors="coerce")
            return numeric.sum(axis=1, min_count=1)

        moca_dict = {
            'executive': _sum_columns(["MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH"]),
            'naming': _sum_columns(["MCALION", "MCARHINO", "MCACAMEL"]),
            'attention_numbers': _sum_columns(["MCAFDS", "MCABDS"]),
            'attention_letters': _sum_columns(["MCAVIGIL"]),
            'attention_substract': _sum_columns(["MCASER7"]),
            'language_rep': _sum_columns(["MCASNTNC"]),
            'language_letters': _sum_columns(["MCAVF"]),
            'abstraction': _sum_columns(["MCAABSTR"]),
            'reminding': _sum_columns(["MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5"]),
            'orientation': _sum_columns(["MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY"]),
            'total': _sum_columns(["MCATOT"])
        }

        moca_dict = dict(zip(covariate_dict['moca'][5:16], moca_dict.values()))
        cd['moca'] = pd.concat([moca.iloc[:, :5], pd.DataFrame(moca_dict), moca.iloc[:, -2:]], axis=1)
        cd['moca'].rename(columns={'INFODT': covariate_dict['moca'][4]}, inplace=True)

        # Replace demo with prepared demo_df
        cd['demo'] = demo_df

        return cd

    # ---------- Helpers operating on provided dicts ----------
    def _augment_moca_scores(self, moca_df: pd.DataFrame) -> pd.DataFrame:
        df = moca_df.copy()
        drop_columns: Set[str] = set()

        for agg_col, item_cols in self.MOCA_CATEGORY_MAP.items():
            present = [col for col in item_cols if col in df.columns]
            if not present:
                continue

            agg_series = df[present].sum(axis=1, skipna=False)
            df[agg_col] = agg_series
            df[f"{agg_col}_sum"] = agg_series

            for col in present:
                if col not in {agg_col, f"{agg_col}_sum"}:
                    drop_columns.add(col)

        if drop_columns:
            df = df.drop(columns=list(drop_columns), errors="ignore")

        base_cols = [col for col in self.MOCA_DATA_COLS if col in df.columns]
        if base_cols:
            df["MoCA_sum"] = df[base_cols].sum(axis=1, skipna=False)

        sum_cols = [f"{col}_sum" for col in self.MOCA_DATA_COLS if f"{col}_sum" in df.columns]
        if sum_cols:
            df["MoCA_sum_post"] = df[sum_cols].sum(axis=1, skipna=False)

        if "MoCA_ONLY_GES" in df.columns:
            existing_sum = df.get("MoCA_sum")
            if existing_sum is not None:
                df["MoCA_sum"] = df["MoCA_ONLY_GES"].combine_first(existing_sum)
            else:
                df["MoCA_sum"] = df["MoCA_ONLY_GES"]
            df.drop(columns=["MoCA_ONLY_GES"], inplace=True, errors="ignore")

        return df

    def _aggregate_custom_moca_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        drop_columns: Set[str] = set()

        suffix_groups = {
            "pre": ["", "_pre"],
            "post": ["_post", "_sum"]
        }
    
        for agg_col, item_cols in self.MOCA_CATEGORY_MAP.items():
            for phase, suffixes in suffix_groups.items():
                target_col = agg_col if phase == "pre" else f"{agg_col}_sum"
                candidates: List[str] = []
                for item in item_cols:
                    for suffix in suffixes:
                        candidate = f"{item}{suffix}" if suffix else item
                        if candidate in result.columns:
                            candidates.append(candidate)
                if not candidates:
                    continue

                agg_series = result[candidates].sum(axis=1, skipna=False)
                result[target_col] = agg_series
                for candidate in candidates:
                    if candidate not in {target_col}:
                        drop_columns.add(candidate)

        if drop_columns:
            result = result.drop(columns=list(drop_columns), errors="ignore")

        base_cols = [col for col in self.MOCA_DATA_COLS if col in result.columns]
        if base_cols:
            result["MoCA_sum"] = result[base_cols].sum(axis=1, skipna=False)

        sum_cols = [f"{col}_sum" for col in self.MOCA_DATA_COLS if f"{col}_sum" in result.columns]
        if sum_cols:
            result["MoCA_sum_post"] = result[sum_cols].sum(axis=1, skipna=False)

        if "MoCA_ONLY_GES" in result.columns:
            existing_sum = result.get("MoCA_sum")
            if existing_sum is not None:
                result["MoCA_sum"] = result["MoCA_ONLY_GES"].combine_first(existing_sum)
            else:
                result["MoCA_sum"] = result["MoCA_ONLY_GES"]
            result.drop(columns=["MoCA_ONLY_GES"], inplace=True, errors="ignore")

        for post_total_col in ("MoCA_ONLY_GES_post", "MoCA_ONLY_GES_sum"):
            if post_total_col in result.columns:
                result["MoCA_sum_post"] = result[post_total_col]
                drop_columns.add(post_total_col)

        if drop_columns:
            result = result.drop(columns=list(drop_columns), errors="ignore")

        return result
    
    def _coalesce_into(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
        """
        If `target` is missing or entirely NaN, copy the first existing candidate column into `target`.
        Case-insensitive column matching. Tries numeric coerce; falls back to raw if non-numeric.
        """
        # Already present and not all-NaN → keep it
        if target in df.columns and not df[target].isna().all():
            return df

        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            src = lower_map.get(cand.lower())
            if src:
                try:
                    df[target] = pd.to_numeric(df[src], errors="coerce")
                    if df[target].isna().all():  # e.g., non-numeric strings
                        df[target] = df[src]
                except Exception:
                    df[target] = df[src]
                break
        return df

    @staticmethod
    def _compute_ledd_pre(
        medication_df: Optional[pd.DataFrame],
        op_dates: pd.Series,
        max_days: int = int(round(365.25 * 2))
    ) -> pd.Series:
        """Return the pre-surgery LEDD measurement closest (but not after) OP_DATUM within the allowed window.

        The default window is roughly two years (±730 days).
        """
        if medication_df is None or medication_df.empty:
            return pd.Series(dtype=float)
        if op_dates is None or op_dates.empty:
            return pd.Series(dtype=float)

        if isinstance(op_dates, pd.Series):
            op_series = op_dates.dropna()
        else:
            return pd.Series(dtype=float)

        if op_series.empty:
            return pd.Series(dtype=float)

        patno_col = next((col for col in medication_df.columns if col.strip().upper() == "PATNO"), None)
        if patno_col is None:
            return pd.Series(dtype=float)

        ledd_candidates = [col for col in medication_df.columns if "LEDD" in col.upper()]
        if not ledd_candidates:
            return pd.Series(dtype=float)
        ledd_col = next((col for col in ledd_candidates if col.upper() == "LEDD"), ledd_candidates[0])

        date_candidates: List[str] = []
        for col in medication_df.columns:
            col_upper = col.upper()
            if any(token in col_upper for token in ("INFODT", "LEDDDT", "RXDTE", "RXDT", "LEDD_DATE", "DATE")):
                date_candidates.append(col)
        if not date_candidates:
            date_candidates = [col for col in medication_df.columns if col.upper().endswith("DT") or "DATE" in col.upper()]
        if not date_candidates:
            return pd.Series(dtype=float)
        date_col = date_candidates[0]

        med = medication_df[[patno_col, date_col, ledd_col]].copy()
        med = med.rename(columns={patno_col: "PATNO", date_col: "MEAS_DATE", ledd_col: "LEDD_VALUE"})
        med = med.dropna(subset=["PATNO"])
        med["LEDD_VALUE"] = pd.to_numeric(med["LEDD_VALUE"], errors="coerce")
        med = med.dropna(subset=["LEDD_VALUE"])
        if med.empty:
            return pd.Series(dtype=float)

        med = safe_parse_dates(med, cols=["MEAS_DATE"], dayfirst=True, report=False)
        med = med.dropna(subset=["MEAS_DATE"])
        if med.empty:
            return pd.Series(dtype=float)

        if pd.api.types.is_datetime64tz_dtype(med["MEAS_DATE"].dtype):
            med["MEAS_DATE"] = med["MEAS_DATE"].dt.tz_localize(None)

        op_series = pd.to_datetime(op_series, errors="coerce").dropna()
        if op_series.empty:
            return pd.Series(dtype=float)

        if pd.api.types.is_datetime64tz_dtype(op_series.dtype):
            op_series = op_series.dt.tz_localize(None)

        merged = med.merge(op_series.rename("OP_DATUM"), left_on="PATNO", right_index=True, how="inner")
        if merged.empty:
            return pd.Series(dtype=float)

        merged = merged[merged["MEAS_DATE"] <= merged["OP_DATUM"]]
        if merged.empty:
            return pd.Series(dtype=float)

        merged["delta_days"] = (merged["OP_DATUM"] - merged["MEAS_DATE"]).dt.days
        merged = merged[(merged["delta_days"] >= 0) & (merged["delta_days"] <= max_days)]
        if merged.empty:
            return pd.Series(dtype=float)

        merged = merged.sort_values(["PATNO", "delta_days", "MEAS_DATE"])
        closest_idx = merged.groupby("PATNO")["delta_days"].idxmin()
        closest = merged.loc[closest_idx, ["PATNO", "LEDD_VALUE"]].drop_duplicates("PATNO", keep="first")
        result = closest.set_index("PATNO")["LEDD_VALUE"]
        result.name = "LEDD_pre"
        return result

    @staticmethod
    def _rename_moca_pre_post(df: pd.DataFrame) -> pd.DataFrame:
        renamed = df.copy()
        rename_map: Dict[str, str] = {}

        for col in renamed.columns:
            if col.startswith("MoCA_"):
                if col.endswith(("_pre", "_post", "_sum_pre", "_sum_post")):
                    continue  # Already standardized
                if col.endswith("_sum"):
                    # assume this is post-treatment, e.g. "MoCA_sum" → "MoCA_sum_post"
                    rename_map[col] = f"{col}_post"
            elif col in {"UPDRS_OFF", "UPDRS_reduc"} and not col.endswith("_pre"):
                rename_map[col] = f"{col}_pre"

        if rename_map:
            renamed = renamed.rename(columns=rename_map)

        return renamed


    def _prepare_moca_with_demo(self, complete_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        moca_df = complete_data.get("moca")
        if moca_df is None:
            return pd.DataFrame()
        moca_df = self._augment_moca_scores(moca_df)
        demo_df = complete_data.get("demo")
        if demo_df is None:
            return moca_df

        demo_keep = ["PATNO","OP_DATUM","LOCATION","DIAG_DATE","SEX","AGE_AT_OP","LEDD_pre"]
        demo_cols = [c for c in demo_keep if c in demo_df.columns]
        demo_subset = demo_df[demo_cols].copy()

        merged = pd.merge(moca_df, demo_subset, on="PATNO", how="inner", suffixes=("", "_demo"))
        for col in demo_cols:
            demo_col = f"{col}_demo"
            if demo_col in merged.columns:
                merged.drop(columns=[col], errors="ignore", inplace=True)
                merged.rename(columns={demo_col: col}, inplace=True)
        return merged

    @staticmethod
    def _numeric_columns(df: pd.DataFrame, exclude: Set[str]) -> List[str]:
        numeric_cols: List[str] = []
        for col in df.columns:
            if col in exclude:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                continue
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().any():
                df[col] = coerced
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
        return numeric_cols

    def _get_updrs3_by_state(self, updrs3: pd.DataFrame, state: str = "off") -> pd.DataFrame:
        """
        Extract MDS-UPDRS Part III totals for a given medication state using PDSTATE when available.
        - ON is: numeric PDSTATE > 0, or strings containing 'ON' (case-insensitive).
        - OFF is: numeric PDSTATE == 0, or strings containing 'OFF'.
        - Fallback to OFFEXAM/ONEXAM if PDSTATE is unavailable or blank.

        Returns a dataframe with columns:
          - PATNO
          - TEST_DATUM (parsed datetime)
          - UPDRS_off or UPDRS_on (depending on 'state')
          - EVENT_ID (included if present in the source)
        """
        if updrs3 is None or updrs3.empty:
            return pd.DataFrame(columns=["PATNO", "TEST_DATUM", f"UPDRS_{state}"])

        state = state.lower().strip()
        if state not in {"off", "on"}:
            raise ValueError("updrs state must be 'off' or 'on'")

        df = updrs3.copy()

        # Identify total score column
        total_candidates = [
            "MDS-UPDRS_3_ONLY_Gesamt",  # mapped name after your renames
            "NP3TOT",
            "UPDRS_III_TOTAL",
            "MDS_UPDRS_III_TOTAL",
        ]
        total_col = next((c for c in total_candidates if c in df.columns), None)
        if total_col is None:
            return pd.DataFrame(columns=["PATNO", "TEST_DATUM", f"UPDRS_{state}"])

        # Identify exam date column
        date_candidates = ["TEST_DATUM", "INFODT", "EXAMDT"]
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col is None:
            return pd.DataFrame(columns=["PATNO", "TEST_DATUM", f"UPDRS_{state}"])

        # ---- Build state mask using PDSTATE, else OFFEXAM/ONEXAM ----
        mask = pd.Series(True, index=df.index)

        if "PDSTATE" in df.columns:
            ps = df["PDSTATE"]
            ps_num = pd.to_numeric(ps, errors="coerce")

            # If any numeric codes exist, use them; else fall back to strings
            if ps_num.notna().any():
                if state == "off":
                    mask = ps_num == 0
                else:
                    mask = ps_num > 0
            else:
                ps_norm = ps.astype(str).str.upper()
                if state == "off":
                    mask = ps_norm.str.contains(r"\bOFF\b", na=False)
                else:
                    mask = ps_norm.str.contains(r"\bON\b", na=False)


        # If PDSTATE didn’t produce a mask (all NA / empty), use OFFEXAM/ONEXAM flags as fallback
        if (mask is None) or (mask.sum() == 0):
            mask = pd.Series(False, index=df.index)
            if state == "off":
                if "OFFEXAM" in df.columns:
                    mask = mask | (df["OFFEXAM"] == 1)
                if "ONEXAM" in df.columns:
                    mask = mask | (df["ONEXAM"] == 0)
            else:
                if "ONEXAM" in df.columns:
                    mask = mask | (df["ONEXAM"] == 1)
                if "OFFEXAM" in df.columns:
                    mask = mask | (df["OFFEXAM"] == 0)

        # Subset and rename
        cols = ["PATNO"]
        if "EVENT_ID" in df.columns:
            cols.append("EVENT_ID")
        cols.extend([date_col, total_col])

        out = df.loc[mask, cols].rename(
            columns={date_col: "TEST_DATUM", total_col: f"UPDRS_{state}"}
        )

        # Clean up and parse dates
        out = out.dropna(subset=["PATNO", "TEST_DATUM", f"UPDRS_{state}"]).copy()
        out = safe_parse_dates(out, cols=["TEST_DATUM"], dayfirst=True, report=False)

        # Return with EVENT_ID if present
        base_cols = ["PATNO", "TEST_DATUM", f"UPDRS_{state}"]
        if "EVENT_ID" in out.columns:
            base_cols.insert(1, "EVENT_ID")

        return out[base_cols]


    def _ledd_nearest_per_visit(
        self,
        medication_df: Optional[pd.DataFrame],
        visits: pd.DataFrame,  # must contain PATNO, TEST_DATUM
        max_days: int = int(round(365.25 * 2)),  # ± ~2 years
        prefer_past: bool = True,
        out_col: str = "LEDD_pre"
    ) -> pd.Series:
        """
        For each (PATNO, TEST_DATUM), return the LEDD value measured nearest to TEST_DATUM.
        Preference is given to measurements BEFORE TEST_DATUM if prefer_past is True; otherwise pure nearest.
        """
        if medication_df is None or medication_df.empty or visits.empty:
            return pd.Series(index=visits.index, dtype=float, name=out_col)

        # Identify columns
        patno_col = next((c for c in medication_df.columns if c.strip().upper() == "PATNO"), None)
        if patno_col is None:
            return pd.Series(index=visits.index, dtype=float, name=out_col)

        ledd_candidates = [col for col in medication_df.columns if "LEDD" in col.upper()]
        if not ledd_candidates:
            return pd.Series(index=visits.index, dtype=float, name=out_col)
        ledd_col = next((c for c in ledd_candidates if c.upper() == "LEDD"), ledd_candidates[0])

        # Date column
        date_candidates = [c for c in medication_df.columns if any(k in c.upper() for k in ("INFODT","LEDDDT","RXDTE","RXDT","LEDD_DATE","DATE"))]
        if not date_candidates:
            date_candidates = [c for c in medication_df.columns if c.upper().endswith("DT") or "DATE" in c.upper()]
        if not date_candidates:
            return pd.Series(index=visits.index, dtype=float, name=out_col)
        date_col = date_candidates[0]

        med = medication_df[[patno_col, date_col, ledd_col]].rename(
            columns={patno_col:"PATNO", date_col:"MEAS_DATE", ledd_col:"LEDD"}
        )
        med = med.dropna(subset=["PATNO"])
        med["LEDD"] = pd.to_numeric(med["LEDD"], errors="coerce")
        med = med.dropna(subset=["LEDD"])
        if med.empty:
            return pd.Series(index=visits.index, dtype=float, name=out_col)

        med = safe_parse_dates(med, cols=["MEAS_DATE"], dayfirst=True, report=False).dropna(subset=["MEAS_DATE"])
        if med.empty:
            return pd.Series(index=visits.index, dtype=float, name=out_col)

        if pd.api.types.is_datetime64tz_dtype(med["MEAS_DATE"].dtype):
            med["MEAS_DATE"] = med["MEAS_DATE"].dt.tz_localize(None)

        V = visits[["PATNO", "TEST_DATUM"]].copy()
        V = safe_parse_dates(V, cols=["TEST_DATUM"], dayfirst=True, report=False).dropna(subset=["TEST_DATUM"])
        V["_row"] = V.index
        if not V.empty and pd.api.types.is_datetime64tz_dtype(V["TEST_DATUM"].dtype):
            V["TEST_DATUM"] = V["TEST_DATUM"].dt.tz_localize(None)

        # Join per patient, compute deltas
        merged = med.merge(V, on="PATNO", how="inner")
        if merged.empty:
            return pd.Series(index=visits.index, dtype=float, name=out_col)

        merged["delta"] = (merged["MEAS_DATE"] - merged["TEST_DATUM"]).dt.days
        merged["delta_abs"] = merged["delta"].abs()
        merged = merged[merged["delta_abs"] <= max_days]
        if merged.empty:
            return pd.Series(index=visits.index, dtype=float, name=out_col)

        # Prefer past when tie-breaking
        merged = merged.sort_values(["_row", "delta_abs", "delta" if prefer_past else "MEAS_DATE"])
        best_idx = merged.groupby("_row")["delta_abs"].idxmin()
        best = merged.loc[best_idx, ["_row", "LEDD"]].drop_duplicates("_row", keep="first")
        out = pd.Series(data=best["LEDD"].values, index=best["_row"].values, name=out_col, dtype=float)
        return out.reindex(visits.index)


    def _build_ppmi_timeline_pairs(
        self,
        df: pd.DataFrame,
        target_years: float = 2.0,
        window_years: float = 2.0,
        min_separation_days: int = 180,
        require_updrs: bool = False,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        vdf = df.copy()
        vdf["TEST_DATUM"] = pd.to_datetime(vdf["TEST_DATUM"], errors="coerce")
        vdf = vdf.dropna(subset=["PATNO", "TEST_DATUM"])

        tgt_days = int(round(target_years * 365.25))
        win_days = int(round(window_years * 365.25))

        rows = []
        for patno, v in vdf.groupby("PATNO"):
            v = v.sort_values("TEST_DATUM")
            if v.shape[0] < 2:
                continue

            # For each FOLLOW-UP, pick the single baseline that makes the gap closest to target
            for i in range(1, len(v)):
                follow = v.iloc[i]
                candidates = v.iloc[:i].copy()

                if require_updrs and "UPDRS_on" in candidates.columns:
                    candidates = candidates[candidates["UPDRS_on"].notna()]
                    if candidates.empty:
                        continue

                candidates["gap_days"] = (follow["TEST_DATUM"] - candidates["TEST_DATUM"]).dt.days
                candidates = candidates[candidates["gap_days"] >= min_separation_days]
                if candidates.empty:
                    continue

                candidates["abs_diff_to_target_days"] = (candidates["gap_days"] - tgt_days).abs()
                candidates = candidates[candidates["abs_diff_to_target_days"] <= win_days]
                if candidates.empty:
                    continue

                base = candidates.sort_values(
                    ["abs_diff_to_target_days", "gap_days"]
                ).iloc[0]

                row = {
                    "PATNO": patno,
                    "TEST_DATUM_pre":  base["TEST_DATUM"],
                    "TEST_DATUM_post": follow["TEST_DATUM"],
                    "TimeSinceBaselineDays": base.get("TimeSinceBaselineDays", np.nan),
                    "TimeSinceDiag":        base.get("TimeSinceDiag", np.nan),   # baseline TSD stays intact
                    "TimeSinceSurgery":     float(base["gap_days"]) / 365.25,
                    "abs_diff_to_target_days": float(base["abs_diff_to_target_days"]),
                    "UPDRS_on": base.get("UPDRS_on", np.nan),
                }

                for c in v.columns:
                    if c.startswith("MoCA_") and c.endswith("_sum"):
                        row[f"{c}_pre"] = base[c]
                        if c in follow:
                            row[f"{c}_post"] = follow[c]
                rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(
                ["PATNO", "abs_diff_to_target_days", "TEST_DATUM_post"]
            ).reset_index(drop=True)
        return out

    @staticmethod
    def _coalesce_visit_date(
        df: pd.DataFrame,
        out_col: str = "TEST_DATUM",
        prefer: Tuple[str, ...] = ("TEST_DATUM", "EXAMDT", "INFODT"),
        dayfirst: bool = True,
    ) -> pd.DataFrame:
        """
        Ensure a visit-level date column exists by copying the first usable candidate
        from `prefer` into `out_col`. Returns a copy so callers can chain safely.
        """
        tmp = df.copy()
        for col in prefer:
            if col in tmp.columns and tmp[col].notna().any():
                parsed = pd.to_datetime(tmp[col], errors="coerce", dayfirst=dayfirst)
                if parsed.notna().any():
                    tmp[out_col] = parsed
                    break
        return tmp

    @staticmethod
    def _standardize_moca_suffixes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize MoCA sub-score suffixes to *_sum_pre/_sum_post so downstream code
        can rely on consistent column names.
        """
        renamed = df.copy()
        rename_map: Dict[str, str] = {}
        for col in renamed.columns:
            if col.startswith("MoCA_"):
                if col.endswith("_sum_pre") or col.endswith("_sum_post"):
                    continue
                if col.endswith("_pre") and not col.endswith("_sum_pre"):
                    rename_map[col] = col.replace("_pre", "_sum_pre")
                elif col.endswith("_post") and not col.endswith("_sum_post"):
                    rename_map[col] = col.replace("_post", "_sum_post")
        if rename_map:
            renamed = renamed.rename(columns=rename_map)
        return renamed

    @staticmethod
    def _non_dbs_keep_columns(
        df: pd.DataFrame,
        who: str,
        id_column: str,
        updrs_col: str,
    ) -> pd.DataFrame:
        """
        Retain core columns used by non-DBS downstream analyses and ensure a COHORT tag.
        """
        keep = [c for c in df.columns if c.startswith("MoCA_")]
        extras = [
            "TimeSinceDiag",
            "TimeSinceSurgery",
            updrs_col,
            updrs_col.upper() if updrs_col.upper() != updrs_col else None,
            "LEDD_pre",
            "AGE_AT_OP",
            "AGE_AT_BASELINE",
            "PATNO",
            id_column,
            "TEST_DATUM",
            "propensity",
            "logit",
            "matched_ppmi_patno",
            "matched_ppmi_index",
            "matched_custom_id",
            "matched_custom_index",
            "cohort",
        ]
        for col in extras:
            if col and col in df.columns and col not in keep:
                keep.append(col)

        keep = list(dict.fromkeys(keep))
        out = df[keep].copy()
        out["COHORT"] = df.get("cohort", who)
        return out

    def _attach_updrs_on_and_off(self, ppmi_visits: pd.DataFrame, ppmi_cd: dict,
                                 nearest_window_days: int = 365, prefer_past: bool = True) -> pd.DataFrame:
        """
        Ensure ppmi_visits has UPDRS_on (as you already do), and then, for rows that have UPDRS_on,
        attach the nearest-in-time UPDRS_off from the same patient within ±nearest_window_days.
        Does NOT require the same EVENT_ID. Prefers past OFF exam in tie situations.
        """
        up3 = ppmi_cd.get("mds_updrs", {}).get("mds_updrs3")
        if up3 is None or up3.empty or ppmi_visits.empty:
            return ppmi_visits

        # --- Pick a robust total column
        total_candidates = [
            "MDS-UPDRS_3_ONLY_Gesamt", "NP3TOT",
            "UPDRS_III_TOTAL", "MDS_UPDRS_III_TOTAL"
        ]
        tot = next((c for c in total_candidates if c in up3.columns), None)
        if tot is None:
            return ppmi_visits

        # --- Ensure a usable exam date
        up3 = self._coalesce_visit_date(up3, out_col="TEST_DATUM",
                                        prefer=("TEST_DATUM", "EXAMDT", "INFODT"))
        up3 = safe_parse_dates(up3, cols=["TEST_DATUM"], dayfirst=True, report=False)
        up3 = up3.dropna(subset=["PATNO", "TEST_DATUM"])

        # --- Build ON/OFF masks
        def _state_mask(df: pd.DataFrame, state: str) -> pd.Series:
            # default all False
            mask = pd.Series(False, index=df.index)

            if "PDSTATE" in df.columns:
                ps_num = pd.to_numeric(df["PDSTATE"], errors="coerce")
                if ps_num.notna().any():
                    if state == "off":
                        mask = ps_num == 0
                    else:
                        mask = ps_num > 0
                else:
                    ps = df["PDSTATE"].astype(str).str.upper()
                    if state == "off":
                        mask = ps.str.contains(r"\bOFF\b", na=False)
                    else:
                        mask = ps.str.contains(r"\bON\b", na=False)

            # fallbacks (only where mask not already True)
            if "OFFEXAM" in df.columns and state == "off":
                mask = mask | (df["OFFEXAM"] == 1)
            if "ONEXAM" in df.columns and state == "on":
                mask = mask | (df["ONEXAM"] == 1)

            return mask.fillna(False)

        mask_on  = _state_mask(up3, "on")
        mask_off = _state_mask(up3, "off")

        on_tbl = up3.loc[mask_on,  ["PATNO", "TEST_DATUM", tot]].rename(columns={tot: "UPDRS_on"}).copy()
        off_tbl = up3.loc[mask_off, ["PATNO", "TEST_DATUM", tot]].rename(columns={tot: "UPDRS_off"}).copy()

        # --- Step 1: keep your current ON logic (EVENT_ID join if present, else nearest).
        # If you already did this earlier, you can skip redoing; otherwise:
        # Try exact merge by (PATNO, EVENT_ID) when both sides have it
        if "EVENT_ID" in ppmi_visits.columns and "EVENT_ID" in up3.columns:
            on_ev = up3.loc[mask_on, ["PATNO", "EVENT_ID", tot]].rename(columns={tot: "UPDRS_on"})
            ppmi_visits = ppmi_visits.merge(on_ev, on=["PATNO", "EVENT_ID"], how="left")

        # Fill remaining ON via nearest-in-time (±nearest_window_days, prefer past)
        need_on = ppmi_visits["UPDRS_on"].isna() if "UPDRS_on" in ppmi_visits.columns else pd.Series(True, index=ppmi_visits.index)
        if need_on.any():
            anchor = ppmi_visits.loc[need_on, ["PATNO", "TEST_DATUM"]].copy()
            anchor = safe_parse_dates(anchor, cols=["TEST_DATUM"], dayfirst=True, report=False).dropna(subset=["TEST_DATUM"])
            if not anchor.empty:
                tmp = anchor.merge(on_tbl.rename(columns={"TEST_DATUM": "ON_TEST_DATUM"}), on="PATNO", how="left")
                if not tmp.empty:
                    tmp["d"] = (tmp["ON_TEST_DATUM"] - tmp["TEST_DATUM"]).dt.days
                    tmp = tmp[tmp["d"].abs() <= int(nearest_window_days)]
                    if not tmp.empty:
                        tmp["abs_d"] = tmp["d"].abs()
                        # prefer past if requested
                        if prefer_past:
                            tmp["past_first"] = (tmp["d"] > 0).astype(int)  # 0 (past) sorts before 1 (future)
                            tmp = (tmp.sort_values(["PATNO","TEST_DATUM","abs_d","past_first"])
                                      .drop_duplicates(subset=["PATNO","TEST_DATUM"], keep="first"))
                        else:
                            tmp = (tmp.sort_values(["PATNO","TEST_DATUM","abs_d"])
                                      .drop_duplicates(subset=["PATNO","TEST_DATUM"], keep="first"))
                        tmp = tmp[["PATNO","TEST_DATUM","UPDRS_on"]]
                        ppmi_visits = ppmi_visits.merge(tmp, on=["PATNO","TEST_DATUM"], how="left", suffixes=("", "_NEARON"))
                        # coalesce
                        if "UPDRS_on" not in ppmi_visits.columns:
                            ppmi_visits["UPDRS_on"] = ppmi_visits["UPDRS_on_NEARON"]
                        else:
                            ppmi_visits["UPDRS_on"] = ppmi_visits["UPDRS_on"].combine_first(ppmi_visits["UPDRS_on_NEARON"])
                        ppmi_visits.drop(columns=[c for c in ppmi_visits.columns if c.endswith("_NEARON")], inplace=True, errors="ignore")

        # --- Step 2: for each ON visit, attach nearest OFF (±nearest_window_days)
        have_on = ppmi_visits["UPDRS_on"].notna() if "UPDRS_on" in ppmi_visits.columns else pd.Series(False, index=ppmi_visits.index)
        anchor = ppmi_visits.loc[have_on, ["PATNO", "TEST_DATUM"]].copy()
        anchor = safe_parse_dates(anchor, cols=["TEST_DATUM"], dayfirst=True, report=False).dropna(subset=["TEST_DATUM"])
        if not anchor.empty and not off_tbl.empty:
            tmp = anchor.merge(off_tbl.rename(columns={"TEST_DATUM": "OFF_TEST_DATUM"}), on="PATNO", how="left")
            if not tmp.empty:
                tmp["d"] = (tmp["OFF_TEST_DATUM"] - tmp["TEST_DATUM"]).dt.days
                tmp = tmp[tmp["d"].abs() <= int(nearest_window_days)]
                if not tmp.empty:
                    tmp["abs_d"] = tmp["d"].abs()
                    if prefer_past:
                        tmp["past_first"] = (tmp["d"] > 0).astype(int)
                        tmp = (tmp.sort_values(["PATNO","TEST_DATUM","abs_d","past_first"])
                                  .drop_duplicates(subset=["PATNO","TEST_DATUM"], keep="first"))
                    else:
                        tmp = (tmp.sort_values(["PATNO","TEST_DATUM","abs_d"])
                                  .drop_duplicates(subset=["PATNO","TEST_DATUM"], keep="first"))
                    tmp = tmp[["PATNO","TEST_DATUM","UPDRS_off"]]
                    ppmi_visits = ppmi_visits.merge(tmp, on=["PATNO","TEST_DATUM"], how="left", suffixes=("", "_NEAROFF"))
                    # coalesce into UPDRS_off (create if missing)
                    if "UPDRS_off" not in ppmi_visits.columns:
                        ppmi_visits["UPDRS_off"] = ppmi_visits["UPDRS_off_NEAROFF"]
                    else:
                        ppmi_visits["UPDRS_off"] = ppmi_visits["UPDRS_off"].combine_first(ppmi_visits["UPDRS_off_NEAROFF"])
                    ppmi_visits.drop(columns=[c for c in ppmi_visits.columns if c.endswith("_NEAROFF")], inplace=True, errors="ignore")

        return ppmi_visits


    def _prepare_non_dbs_data(
        self,
        csv_path: str,
        quest: str,
        id_column: str,
        use_updrs: bool,
        updrs_state: str,
    ) ->     Dict[str, pd.DataFrame | str]:
        """
        Shared preprocessing pipeline for non-DBS analyses.
        Returns aligned custom/PT and PPMI cohorts ready for downstream selection.

        Guarantees:
        - High coverage for UPDRS_on (legacy behavior).
        - Adds UPDRS_off additively (when requested), without disturbing ON.
        - Never leaves *_x/*_y/*_NEAR artifacts behind.
        """
        state = updrs_state.lower().strip()
        if state not in {"off", "on", "both"}:
            raise ValueError("updrs_state must be 'off', 'on', or 'both'")
        updrs_col = "UPDRS_on" if state in {"on", "both"} else "UPDRS_off"

        # ---------- helpers (safe combining & cleanup) ----------
        def _combine_into(df: pd.DataFrame, base: str, temp: str) -> None:
            """Ensure df[base] exists, then combine_first from df[temp] and drop temp."""
            if temp not in df.columns:
                return
            if base not in df.columns:
                df[base] = np.nan
            df[base] = df[base].combine_first(df[temp])
            df.drop(columns=[temp], inplace=True, errors="ignore")

        def _collapse_score(df: pd.DataFrame, base: str) -> pd.DataFrame:
            """Fold any variants into `base` then drop suffix columns."""
            # Fold *_NEAR first if present
            if f"{base}_NEAR" in df.columns:
                _combine_into(df, base, f"{base}_NEAR")
            # Fold any *_ADD, *_x, *_y leftover
            for c in list(df.columns):
                if c == base:
                    continue
                if c.startswith(base + "_") or c.endswith("_x") or c.endswith("_y"):
                    # If it's a direct variant (e.g., base_ADD), fold; else just drop
                    if c.startswith(base + "_"):
                        _combine_into(df, base, c)
                    else:
                        df.drop(columns=[c], inplace=True, errors="ignore")
            return df

        # ---------- CUSTOM ----------
        custom_df = pd.read_csv(csv_path)

        # normalize custom UPDRS names and MoCA
        columns_to_rename = {
            c: c.replace("_pre", "") for c in custom_df.columns
            if c.startswith("UPDRS") and c.endswith("_pre")
        }
        if columns_to_rename:
            custom_df = custom_df.rename(columns=columns_to_rename)
        custom_df = custom_df.rename(columns={"MDS_UPDRS_III_sum_ON": "UPDRS_ON"})
        custom_df = self._aggregate_custom_moca_columns(custom_df)
        if "UPDRS_ON" in custom_df.columns and "UPDRS_on" not in custom_df.columns:
            custom_df["UPDRS_on"] = custom_df["UPDRS_ON"]

        # TimeSinceDiag convenience
        if "TimeSinceDiagYears" in custom_df.columns and "TimeSinceDiag" not in custom_df.columns:
            custom_df["TimeSinceDiag"] = custom_df["TimeSinceDiagYears"]
        if "TimeSinceDiagYears" in custom_df.columns and "TimeSinceDiag" in custom_df.columns:
            custom_df.drop(columns=["TimeSinceDiagYears"], inplace=True)

        # Visit date (TEST_DATUM)
        custom_df = self._coalesce_visit_date(custom_df, out_col="TEST_DATUM",
                                              prefer=("TEST_DATUM", "EXAMDT", "INFODT"))
        if "TEST_DATUM" not in custom_df.columns or custom_df["TEST_DATUM"].isna().all():
            if "OP_DATUM" in custom_df.columns and "TimeSinceSurgery" in custom_df.columns:
                custom_df["OP_DATUM"] = pd.to_datetime(custom_df["OP_DATUM"], errors="coerce")
                custom_df["TEST_DATUM"] = custom_df.apply(
                    lambda r: r["OP_DATUM"] + pd.DateOffset(
                        days=int(pd.to_numeric(r.get("TimeSinceSurgery"), errors="coerce") * 365.25)
                    )
                    if pd.notna(r.get("OP_DATUM")) and
                       pd.notna(pd.to_numeric(r.get("TimeSinceSurgery"), errors="coerce"))
                    else pd.NaT,
                    axis=1,
                )

        custom_df.rename(columns={"Pat_ID": "PATNO"}, inplace=True)
        custom_df = safe_parse_dates(custom_df,
                                     cols=[c for c in ("TEST_DATUM", "OP_DATUM") if c in custom_df.columns],
                                     dayfirst=True,
                                     report=False)
        if id_column not in custom_df.columns:
            raise ValueError("Provide id_column for the custom cohort.")
        custom_df = custom_df.dropna(subset=[id_column, "TEST_DATUM"]).copy()
        custom_df[id_column] = custom_df[id_column].astype(str)
        custom_df.sort_values([id_column, "TEST_DATUM"], inplace=True)

        first_test = custom_df.groupby(id_column)["TEST_DATUM"].transform("min")
        custom_df["TimeSinceBaselineDays"] = (custom_df["TEST_DATUM"] - first_test).dt.days
        custom_df["TimeSinceBaselineYears"] = custom_df["TimeSinceBaselineDays"] / 365.25

        if "LEDD_pre" not in custom_df.columns:
            custom_df["LEDD_pre"] = np.nan

        if "AGE_AT_BASELINE" in custom_df.columns:
            custom_df["AGE_AT_BASELINE"] = pd.to_numeric(custom_df["AGE_AT_BASELINE"], errors="coerce")
        else:
            age_like = [c for c in custom_df.columns if c.upper() in {"AGE", "AGE_AT_OP", "AGE_AT_BASELINE"}]
            if age_like:
                custom_df["AGE_AT_BASELINE"] = pd.to_numeric(custom_df[age_like[0]], errors="coerce")
            elif "BIRTHDT" in custom_df.columns:
                custom_df = safe_parse_dates(custom_df, cols=["BIRTHDT"], dayfirst=True, report=False)
                custom_df["AGE_AT_BASELINE"] = (custom_df["TEST_DATUM"] - custom_df["BIRTHDT"]).dt.days / 365.25
            else:
                custom_df["AGE_AT_BASELINE"] = np.nan

        if quest.lower() != "moca":
            raise NotImplementedError("Non-DBS matching currently supports quest='moca' only.")

        # ---------- PPMI ----------
        raw = self.load_ppmi()
        ppmi_cd = self.convert_to_standard_keys(raw, DBS=False)

        ppmi_df = self._prepare_moca_with_demo(ppmi_cd)
        if ppmi_df.empty:
            raise ValueError("PPMI MoCA dataset is empty after preprocessing.")
        ppmi_df = self._coalesce_visit_date(ppmi_df, out_col="TEST_DATUM",
                                            prefer=("TEST_DATUM", "EXAMDT", "INFODT"))

        # exclude DBS patnos
        dbs_df = ppmi_cd.get("dbs")
        if dbs_df is not None and "PATNO" in dbs_df.columns:
            ppmi_df = ppmi_df[~ppmi_df["PATNO"].isin(set(dbs_df["PATNO"].unique()))]

        ppmi_df = safe_parse_dates(ppmi_df, cols=["TEST_DATUM", "DIAG_DATE"], dayfirst=True, report=False)
        ppmi_df = ppmi_df.dropna(subset=["PATNO", "TEST_DATUM"]).copy()
        ppmi_df.sort_values(["PATNO", "TEST_DATUM"], inplace=True)

        ppmi_first = ppmi_df.groupby("PATNO")["TEST_DATUM"].transform("min")
        ppmi_df["TimeSinceBaselineDays"] = (ppmi_df["TEST_DATUM"] - ppmi_first).dt.days
        ppmi_df["TimeSinceBaselineYears"] = ppmi_df["TimeSinceBaselineDays"] / 365.25

        if "DIAG_DATE" in ppmi_df.columns:
            ppmi_df["TimeSinceDiag"] = ((ppmi_df["TEST_DATUM"] - ppmi_df["DIAG_DATE"]).dt.days / 365.25)
            by_pat = ppmi_df.groupby("PATNO")["TimeSinceDiag"].transform(lambda s: s.dropna().min())
            ppmi_df["TimeSinceDiag"] = ppmi_df["TimeSinceDiag"].fillna(by_pat)

        # LEDD near visit
        ppmi_df["LEDD_pre"] = self._ledd_nearest_per_visit(
            medication_df=ppmi_cd.get("medication"),
            visits=ppmi_df[["PATNO", "TEST_DATUM"]],
            max_days=int(round(365.25 * 2)),
            prefer_past=True,
            out_col="LEDD_pre",
        )

        # Age at visit
        raw_demo = raw.get("demo", pd.DataFrame())
        if not raw_demo.empty and "BIRTHDT" in raw_demo.columns and "PATNO" in raw_demo.columns:
            raw_demo = safe_parse_dates(raw_demo, cols=["BIRTHDT"], dayfirst=True, report=False)
            base_age = raw_demo[["PATNO", "BIRTHDT"]].dropna()
            ppmi_df = ppmi_df.merge(base_age, on="PATNO", how="left")
            ppmi_df["AGE_AT_BASELINE"] = (ppmi_df["TEST_DATUM"] - ppmi_df["BIRTHDT"]).dt.days / 365.25
            ppmi_df.drop(columns=["BIRTHDT"], inplace=True, errors="ignore")
        else:
            ppmi_df["AGE_AT_BASELINE"] = np.nan

        # ---------- UPDRS attach (ON is primary; OFF is additive if requested) ----------
        if use_updrs and "mds_updrs" in ppmi_cd:
            up3 = ppmi_cd["mds_updrs"].get("mds_updrs3")
            if up3 is not None and not up3.empty:
                up3 = self._coalesce_visit_date(up3, out_col="TEST_DATUM",
                                                prefer=("TEST_DATUM", "EXAMDT", "INFODT"))
                up3 = safe_parse_dates(up3, cols=["TEST_DATUM"], dayfirst=True, report=False)

                # ---- ON (preserve legacy coverage) ----
                on_tbl = self._get_updrs3_by_state(up3, state="on")
                if not on_tbl.empty:
                    # A) EVENT_ID exact merge into temp then fold
                    if "EVENT_ID" in ppmi_df.columns and "EVENT_ID" in on_tbl.columns:
                        add = on_tbl[["PATNO", "EVENT_ID", "UPDRS_on"]].rename(columns={"UPDRS_on": "UPDRS_on_ADD"})
                        ppmi_df = ppmi_df.merge(add, on=["PATNO", "EVENT_ID"], how="left")
                        _combine_into(ppmi_df, "UPDRS_on", "UPDRS_on_ADD")

                    # B) nearest ±365d fill
                    need = ppmi_df[["PATNO", "TEST_DATUM"]].copy()
                    tmp = need.merge(on_tbl.rename(columns={"TEST_DATUM": "MEAS_DATE"}), on="PATNO", how="left")
                    if not tmp.empty:
                        tmp["d"] = (pd.to_datetime(tmp["MEAS_DATE"]) - pd.to_datetime(tmp["TEST_DATUM"])).dt.days
                        tmp = tmp[tmp["d"].abs() <= 365]
                        if not tmp.empty:
                            tmp["abs_d"] = tmp["d"].abs()
                            tmp["past_first"] = (tmp["d"] > 0).astype(int)
                            tmp = (tmp.sort_values(["PATNO", "TEST_DATUM", "abs_d", "past_first"])
                                      .drop_duplicates(subset=["PATNO", "TEST_DATUM"], keep="first"))
                            tmp = tmp.rename(columns={"UPDRS_on": "UPDRS_on_NEAR"})
                            ppmi_df = ppmi_df.merge(tmp[["PATNO", "TEST_DATUM", "UPDRS_on_NEAR"]],
                                                    on=["PATNO", "TEST_DATUM"], how="left")
                            _combine_into(ppmi_df, "UPDRS_on", "UPDRS_on_NEAR")

                    _collapse_score(ppmi_df, "UPDRS_on")

                # ---- OFF (optional/additive) ----
                if state in {"off", "both"}:
                    off_tbl = self._get_updrs3_by_state(up3, state="off")
                    if not off_tbl.empty:
                        if "EVENT_ID" in ppmi_df.columns and "EVENT_ID" in off_tbl.columns:
                            add = off_tbl[["PATNO", "EVENT_ID", "UPDRS_off"]].rename(columns={"UPDRS_off": "UPDRS_off_ADD"})
                            ppmi_df = ppmi_df.merge(add, on=["PATNO", "EVENT_ID"], how="left")
                            _combine_into(ppmi_df, "UPDRS_off", "UPDRS_off_ADD")

                        need = ppmi_df[["PATNO", "TEST_DATUM"]].copy()
                        tmp = need.merge(off_tbl.rename(columns={"TEST_DATUM": "MEAS_DATE"}), on="PATNO", how="left")
                        if not tmp.empty:
                            tmp["d"] = (pd.to_datetime(tmp["MEAS_DATE"]) - pd.to_datetime(tmp["TEST_DATUM"])).dt.days
                            tmp = tmp[tmp["d"].abs() <= 365]
                            if not tmp.empty:
                                tmp["abs_d"] = tmp["d"].abs()
                                tmp["past_first"] = (tmp["d"] > 0).astype(int)
                                tmp = (tmp.sort_values(["PATNO", "TEST_DATUM", "abs_d", "past_first"])
                                          .drop_duplicates(subset=["PATNO", "TEST_DATUM"], keep="first"))
                                tmp = tmp.rename(columns={"UPDRS_off": "UPDRS_off_NEAR"})
                                ppmi_df = ppmi_df.merge(tmp[["PATNO", "TEST_DATUM", "UPDRS_off_NEAR"]],
                                                        on=["PATNO", "TEST_DATUM"], how="left")
                                _combine_into(ppmi_df, "UPDRS_off", "UPDRS_off_NEAR")

                    _collapse_score(ppmi_df, "UPDRS_off")

        # Debug coverage
        if "UPDRS_on" in ppmi_df.columns:
            n_on = ppmi_df["UPDRS_on"].notna().sum()
            print(f"[DEBUG] PPMI visits: {len(ppmi_df)} | UPDRS ON attached: {n_on} ({n_on/len(ppmi_df):.2%})")
        if "UPDRS_off" in ppmi_df.columns:
            n_off = ppmi_df["UPDRS_off"].notna().sum()
            print(f"[DEBUG] PPMI visits: {len(ppmi_df)} | UPDRS OFF attached: {n_off} ({n_off/len(ppmi_df):.2%})")

        # ---------- Optional state coverage filter ----------
        if use_updrs:
            need_col = "UPDRS_on" if state in {"on", "both"} else "UPDRS_off"
            if need_col not in ppmi_df.columns:
                raise ValueError(f"{need_col} column is missing after attachment – cannot proceed.")
            ppmi_df["has_state"] = ppmi_df[need_col].notna()
            coverage = ppmi_df.groupby("PATNO")["has_state"].mean()
            eligible_patnos = coverage[coverage >= 0.50].index
            ppmi_df = ppmi_df[ppmi_df["PATNO"].isin(eligible_patnos) & ppmi_df["has_state"]].copy()
            ppmi_df.drop(columns=["has_state"], inplace=True, errors="ignore")

            if need_col in custom_df.columns:
                if pd.api.types.is_bool_dtype(custom_df[need_col]):
                    custom_df = custom_df[custom_df[need_col]].copy()
                else:
                    custom_df = custom_df[custom_df[need_col].notna()].copy()

            print(f"[DEBUG] {need_col.upper()} filter: eligible PPMI patients ≥50% {need_col.upper().split('_')[-1]}: "
                  f"{len(eligible_patnos)} | visits kept: {len(ppmi_df)} | "
                  f"mean coverage among eligibles: {coverage.loc[eligible_patnos].mean():.2%}")

        # ---------- Build pair table and carry covariates ----------
        ppmi_df_pairs = self._build_ppmi_timeline_pairs(ppmi_df)
        ppmi_model = ppmi_df_pairs.copy()

        base_cols = ["PATNO", "TEST_DATUM", "LEDD_pre", "AGE_AT_BASELINE"]
        if "UPDRS_on"  in ppmi_df.columns: base_cols.append("UPDRS_on")
        if "UPDRS_off" in ppmi_df.columns: base_cols.append("UPDRS_off")

        base_cov = ppmi_df[base_cols].drop_duplicates()
        ppmi_model = ppmi_model.merge(
            base_cov.rename(columns={"TEST_DATUM": "TEST_DATUM_pre"}),
            on=["PATNO", "TEST_DATUM_pre"],
            how="left",
        )
        ppmi_model = ppmi_model.rename(columns={"TEST_DATUM_post": "TEST_DATUM"})

        # ---------- Custom model ----------
        custom_model = custom_df.dropna(subset=["TimeSinceBaselineDays"]).copy()
        if custom_model.empty or ppmi_model.empty:
            raise ValueError("No rows with usable timing in one of the cohorts.")

        # Optional MoCA completeness filter
        if "MoCA_sum_pre" in custom_model.columns and "MoCA_sum_pre" in ppmi_model.columns:
            pre_n_custom = len(custom_model)
            pre_n_ppmi = len(ppmi_model)
            custom_model = custom_model.dropna(subset=["MoCA_sum_pre"]).copy()
            ppmi_model = ppmi_model.dropna(subset=["MoCA_sum_pre"]).copy()
            print(f"[PRIORITY] Dropped {pre_n_custom - len(custom_model)} custom and "
                  f"{pre_n_ppmi - len(ppmi_model)} PPMI rows with missing MoCA_sum_pre before matching.")
        else:
            print("[PRIORITY] MoCA_sum_pre not present on one side; skipping MoCA completeness filter.")

        # STRICT: require TimeSinceDiag
        custom_model = custom_model.dropna(subset=["TimeSinceDiag"]).copy()
        ppmi_model   = ppmi_model.dropna(subset=["TimeSinceDiag"]).copy()
        if custom_model.empty or ppmi_model.empty:
            raise ValueError("After enforcing non-missing TimeSinceDiag, one cohort is empty.")

        return {
            "custom_df": custom_df,
            "custom_model": custom_model,
            "ppmi_model": ppmi_model,
            "updrs_col": updrs_col,
            "updrs_state": state,
        }








    
    def select_medication_distribution_cohort(
        self,
        csv_path: str,
        quest: str,
        id_column: str,
        soft_quota_tolerance: float = 0.30,
        *,
        use_updrs: bool = True,
        updrs_state: str = "on",            # now accepts: "on", "off", or "both"
        unique_patient: bool = True,        # enforce ≤1 row per PATNO (recommended)
        require_ledd: bool = False,         # do NOT require LEDD by default
        n_quantile_bins_per_dim: int = 6,   # auto-coarsens if infeasible
        support_trim_quantiles: tuple[float, float] = (0.02, 0.98),  # kept for API compat; not used now
        random_state: int | None = 42,
        ) -> Dict[str, pd.DataFrame]:
        """
        Build a PPMI-only cohort (no 1:1 pairs) whose joint distribution across:
          - TimeSinceSurgery
          - TimeSinceDiag
          - AGE_MATCH  (custom: AGE_AT_OP, ppmi: AGE_AT_BASELINE@TEST_DATUM_pre)

        mirrors the custom cohort. Uses the full pair-level candidate pool first, then enforces
        unique-patient selection while filling per-cell quotas (from custom), and finally
        reallocates remaining PATNOs into any bins with headroom to maximize N.

        NEW: updrs_state="both" attaches both UPDRS_on and UPDRS_off to the PPMI candidate rows
             (and carries through to the sampled cohort). If only one is available, the other is NaN.
        """
        rng = np.random.default_rng(random_state)

        # ---------- 1) Prepare data (PAIR LEVEL) ----------
        # Handle UPDRS state logic, including "both"
        if updrs_state.lower() == "both":
            # Build ON and OFF pools separately, then merge OFF scores into the ON pool by (PATNO, TEST_DATUM_pre, TEST_DATUM)
            prep_on  = self._prepare_non_dbs_data(
                csv_path=csv_path, quest=quest, id_column=id_column,
                use_updrs=use_updrs, updrs_state="on",
            )
            prep_off = self._prepare_non_dbs_data(
                csv_path=csv_path, quest=quest, id_column=id_column,
                use_updrs=use_updrs, updrs_state="off",
            )

            custom = prep_on["custom_model"].copy()   # same custom timing base
            ppmi_on  = prep_on["ppmi_model"].copy()
            ppmi_off = prep_off["ppmi_model"].copy()

            # Ensure merge keys exist
            for df in (ppmi_on, ppmi_off):
                if "TEST_DATUM" not in df.columns and "TEST_DATUM_post" in df.columns:
                    df = df.rename(columns={"TEST_DATUM_post": "TEST_DATUM"})
            # Keep only needed columns from the OFF table to avoid column clutter
            off_keep = ["PATNO", "TEST_DATUM_pre", "TEST_DATUM"]
            if "UPDRS_off" in ppmi_off.columns:
                off_keep.append("UPDRS_off")
            ppmi_off_min = ppmi_off[off_keep].drop_duplicates()

            # Merge OFF scores into ON pool (row-wise alignment: same PATNO and same pair timing)
            ppmi = ppmi_on.merge(ppmi_off_min, on=["PATNO", "TEST_DATUM_pre", "TEST_DATUM"], how="left", suffixes=("", "_offdup"))

            # Make sure UPDRS_on is present (from ON pool). If missing, leave as NaN.
            if "UPDRS_on" not in ppmi.columns and "UPDRS_ON" in ppmi.columns:
                ppmi.rename(columns={"UPDRS_ON": "UPDRS_on"}, inplace=True)

            # For safety, if ON pool had UPDRS_off too, prefer the one from the OFF merge
            if "UPDRS_off_offdup" in ppmi.columns and "UPDRS_off" not in ppmi.columns:
                ppmi.rename(columns={"UPDRS_off_offdup": "UPDRS_off"}, inplace=True)
            elif "UPDRS_off_offdup" in ppmi.columns:
                # combine_first to prefer merged OFF values
                ppmi["UPDRS_off"] = ppmi["UPDRS_off_offdup"].combine_first(ppmi.get("UPDRS_off"))
                ppmi.drop(columns=["UPDRS_off_offdup"], errors="ignore", inplace=True)

        else:
            prep = self._prepare_non_dbs_data(
                csv_path=csv_path,
                quest=quest,
                id_column=id_column,
                use_updrs=use_updrs,
                updrs_state=updrs_state,
            )
            custom = prep["custom_model"].copy()
            ppmi   = prep["ppmi_model"].copy()

        if custom.empty or ppmi.empty:
            raise ValueError("Custom or PPMI pool is empty after preprocessing.")

        # --- Build aligned features ---
        for df, col in ((custom, "TimeSinceSurgery"),
                        (ppmi,   "TimeSinceSurgery"),
                        (custom, "TimeSinceDiag"),
                        (ppmi,   "TimeSinceDiag")):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # AGE alignment: custom uses AGE_AT_OP; ppmi uses AGE_AT_BASELINE (age at TEST_DATUM_pre)
        def _pick_numeric(df: pd.DataFrame, preferred: str, fallback: str | None = None) -> pd.Series:
            if preferred in df.columns:
                return pd.to_numeric(df[preferred], errors="coerce")
            if fallback and fallback in df.columns:
                return pd.to_numeric(df[fallback], errors="coerce")
            return pd.Series(np.nan, index=df.index)

        custom["AGE_MATCH"] = _pick_numeric(custom, "AGE_AT_OP", "AGE_AT_BASELINE")
        ppmi["AGE_MATCH"]   = _pick_numeric(ppmi,   "AGE_AT_BASELINE", "AGE_AT_OP")

        # Optional LEDD requirement
        if require_ledd:
            if "LEDD_pre" not in custom.columns or "LEDD_pre" not in ppmi.columns:
                raise ValueError("LEDD_pre missing. Set require_ledd=False to skip.")
            custom = custom[custom["LEDD_pre"].notna()]
            ppmi   = ppmi[ppmi["LEDD_pre"].notna()]
            if custom.empty or ppmi.empty:
                raise ValueError("After enforcing LEDD_pre, one cohort is empty.")
        else:
            if "LEDD_pre" not in custom.columns: custom["LEDD_pre"] = np.nan
            if "LEDD_pre" not in ppmi.columns:   ppmi["LEDD_pre"]   = np.nan

        # Require complete features (rowwise); **no joint 3D support trim**
        feats = ("TimeSinceSurgery", "TimeSinceDiag", "AGE_MATCH")
        custom = custom.dropna(subset=list(feats)).copy()
        ppmi   = ppmi.dropna(subset=list(feats)).copy()
        if custom.empty or ppmi.empty:
            return {"ppmi": ppmi.iloc[0:0].copy(), "bin_plan": pd.DataFrame(), "balance": pd.DataFrame()}

        # ---------- helpers ----------
        def _edges(x: pd.Series, k: int) -> np.ndarray:
            qs = np.linspace(0, 1, k + 1)
            arr = np.unique(x.quantile(qs, interpolation="linear").to_numpy())
            if len(arr) < 2:
                arr = np.array([float(x.min()), float(x.max())])
            return arr

        def _best_row(rows: list[int]) -> int:
            """Pick a single pair-row for a PATNO within a cell (closest to 2y gap if available)."""
            if not rows:
                return None
            if "abs_diff_to_target_days" in ppmi.columns:
                return min(rows, key=lambda r: (float(ppmi.at[r, "abs_diff_to_target_days"]), ppmi.at[r, "TEST_DATUM"]))
            return min(rows, key=lambda r: ppmi.at[r, "TEST_DATUM"])

        def _smd(a: pd.Series, b: pd.Series) -> float:
            a = pd.to_numeric(a, errors="coerce").dropna()
            b = pd.to_numeric(b, errors="coerce").dropna()
            if len(a) < 2 or len(b) < 2: return np.nan
            m1, m0 = a.mean(), b.mean()
            s1, s0 = a.std(ddof=1), b.std(ddof=1)
            den = np.sqrt((s1**2 + s0**2) / 2.0)
            return float((m1 - m0) / den) if den > 0 else np.nan

        # ---------- 2) Auto-coarsen until feasible unique-PATNO assignment ----------
        bins = int(n_quantile_bins_per_dim)
        while bins >= 2:
            # Bin edges from CUSTOM only
            e1 = _edges(custom["TimeSinceSurgery"], bins)
            e2 = _edges(custom["TimeSinceDiag"],    bins)
            e3 = _edges(custom["AGE_MATCH"],        bins)

            C1 = pd.cut(custom["TimeSinceSurgery"], e1, include_lowest=True)
            C2 = pd.cut(custom["TimeSinceDiag"],    e2, include_lowest=True)
            C3 = pd.cut(custom["AGE_MATCH"],        e3, include_lowest=True)
            P1 = pd.cut(ppmi["TimeSinceSurgery"],   e1, include_lowest=True)
            P2 = pd.cut(ppmi["TimeSinceDiag"],      e2, include_lowest=True)
            P3 = pd.cut(ppmi["AGE_MATCH"],          e3, include_lowest=True)

            C_cells = pd.MultiIndex.from_frame(pd.DataFrame({
                "TimeSinceSurgery": C1, "TimeSinceDiag": C2, "AGE_MATCH": C3
            }))
            P_cells = pd.MultiIndex.from_frame(pd.DataFrame({
                "TimeSinceSurgery": P1, "TimeSinceDiag": P2, "AGE_MATCH": P3
            }))

            # Counts
            C_counts = pd.Series(1, index=custom.index).groupby(C_cells, observed=False).sum()
            P_counts = pd.Series(1, index=ppmi.index).groupby(P_cells, observed=False).sum()

            # Keep cells that exist in CUSTOM and have PPMI availability
            all_cells = C_counts.index
            C_counts = C_counts.reindex(all_cells, fill_value=0)
            P_counts = P_counts.reindex(all_cells, fill_value=0)
            supported = P_counts > 0
            C_pos = C_counts[supported]
            P_pos = P_counts[supported]
            if C_pos.sum() == 0 or P_pos.sum() == 0:
                bins -= 1
                continue

            # Custom proportions over supported cells
            p = (C_pos / int(C_pos.sum())).astype(float)

            # Distinct PATNO capacity per cell
            tmp = pd.DataFrame({"cell": P_cells, "PATNO": ppmi["PATNO"]})
            tmp = tmp[supported.reindex(P_cells, fill_value=False).values]
            distinct_by_cell = (
                tmp.groupby("cell", observed=False)["PATNO"]
                   .nunique()
                   .reindex(p.index, fill_value=0)
            )

            # Overall unique PATNO budget
            PATNO_pool = set(tmp["PATNO"].astype(str).unique())
            N_total = len(PATNO_pool)
            if N_total == 0:
                bins -= 1
                continue

            # Base per-cell quotas from proportions, capped by distinct capacity
            desired = (p * N_total).astype(float)
            n_c = np.floor(desired).astype(int)
            rem = N_total - int(n_c.sum())
            if rem > 0:
                frac = (desired - n_c).sort_values(ascending=False)
                for cell in frac.index:
                    if rem <= 0:
                        break
                    if n_c[cell] < int(distinct_by_cell[cell]):
                        n_c[cell] += 1
                        rem -= 1
            quota = np.minimum(n_c, distinct_by_cell).astype(int)
            if quota.sum() == 0:
                bins -= 1
                continue

            # ---------- 3) Greedy unique-PATNO assignment honoring quotas ----------
            # Build maps
            ppmi_idx = ppmi.index.to_numpy()
            rows_by_cell: Dict[tuple, list[int]] = {}
            for idx, cell in zip(ppmi_idx, P_cells):
                if cell in quota.index:
                    rows_by_cell.setdefault(cell, []).append(int(idx))

            # per PATNO, list rows per cell
            rows_by_patno: Dict[str, Dict[tuple, list[int]]] = {}
            for cell, rows in rows_by_cell.items():
                for r in rows:
                    pat = str(ppmi.at[r, "PATNO"])
                    rows_by_patno.setdefault(pat, {}).setdefault(cell, []).append(r)

            # Best row per (PATNO, cell)
            best_row_for_pat_cell: Dict[tuple, int] = {}
            for pat, cell_map in rows_by_patno.items():
                for cell, rlist in cell_map.items():
                    best_row_for_pat_cell[(pat, cell)] = _best_row(rlist)

            taken_patnos: Set[str] = set()
            chosen_rows: Dict[str, int] = {}
            fill: Dict[tuple, int] = {cell: 0 for cell in quota.index}

            # Fill tightest cells first (fewest distinct patnos)
            cell_order = sorted(quota.index, key=lambda c: (distinct_by_cell[c], quota[c]))
            for cell in cell_order:
                need = int(quota[cell])
                if need <= 0:
                    continue
                # PATNOs that can supply this cell and are not yet taken
                pats = [pat for pat in PATNO_pool if (pat not in taken_patnos) and ((pat, cell) in best_row_for_pat_cell)]
                rng.shuffle(pats)
                for pat in pats:
                    if fill[cell] >= need:
                        break
                    r = best_row_for_pat_cell[(pat, cell)]
                    if r is None:
                        continue
                    chosen_rows[pat] = r
                    taken_patnos.add(pat)
                    fill[cell] += 1

            # ---------- 4) Reallocation: use remaining PATNOs anywhere with soft headroom ----------
            soft_ceiling = np.floor(np.minimum(distinct_by_cell, desired * (1.0 + soft_quota_tolerance))).astype(int)

            remaining = [pat for pat in PATNO_pool if pat not in taken_patnos]
            rng.shuffle(remaining)

            def headroom(c): return int(soft_ceiling.get(c, 0)) - int(fill.get(c, 0))

            while remaining:
                any_added = False
                # iterate cells with max headroom first
                for cell in sorted(soft_ceiling.index, key=headroom, reverse=True):
                    if headroom(cell) <= 0:
                        continue
                    # find a remaining PATNO that can supply this cell
                    picked = None
                    for i, pat in enumerate(remaining):
                        if (pat, cell) in best_row_for_pat_cell:
                            picked = (i, pat)
                            break
                    if picked is None:
                        continue
                    i, pat = picked
                    r = best_row_for_pat_cell[(pat, cell)]
                    if r is None:
                        remaining.pop(i)
                        continue
                    chosen_rows[pat] = r
                    taken_patnos.add(pat)
                    fill[cell] += 1
                    remaining.pop(i)
                    any_added = True
                if not any_added:
                    break

            if len(chosen_rows) == 0:
                bins -= 1
                continue

            # ---------- 5) Outputs ----------
            sampled_ppmi = ppmi.loc[list(chosen_rows.values())].copy().reset_index(drop=True)

            # Plan
            plan_rows = []
            for cell in quota.index:
                plan_rows.append({
                    "cell": cell,
                    "target_quota": int(quota[cell]),
                    "soft_ceiling": int(soft_ceiling[cell]),
                    "filled": int(fill[cell]),
                    "ppmi_distinct_patnos": int(distinct_by_cell[cell]),
                    "ppmi_rows": int(P_pos.get(cell, 0)),
                    "custom_prop": float(p.get(cell, 0.0)),
                })
            bin_plan = pd.DataFrame(plan_rows).sort_values(["filled", "target_quota"], ascending=[False, False])

            # Balance (use aligned AGE_MATCH for both sides)
            def _smd_local(a, b):  # small wrapper avoids shadowing outer
                return _smd(a, b)

            bal = pd.DataFrame({
                "custom_mean": [custom["TimeSinceSurgery"].mean(),
                                custom["TimeSinceDiag"].mean(),
                                custom["AGE_MATCH"].mean()],
                "ppmi_mean":   [sampled_ppmi["TimeSinceSurgery"].mean(),
                                sampled_ppmi["TimeSinceDiag"].mean(),
                                sampled_ppmi["AGE_MATCH"].mean()],
                "pre_SMD":     [_smd_local(custom["TimeSinceSurgery"], ppmi["TimeSinceSurgery"]),
                                _smd_local(custom["TimeSinceDiag"],    ppmi["TimeSinceDiag"]),
                                _smd_local(custom["AGE_MATCH"],        ppmi["AGE_MATCH"])],
                "post_SMD":    [_smd_local(custom["TimeSinceSurgery"], sampled_ppmi["TimeSinceSurgery"]),
                                _smd_local(custom["TimeSinceDiag"],    sampled_ppmi["TimeSinceDiag"]),
                                _smd_local(custom["AGE_MATCH"],        sampled_ppmi["AGE_MATCH"])],
            }, index=["TimeSinceSurgery","TimeSinceDiag","AGE_MATCH"])

            sampled_ppmi = self._rename_moca_pre_post(sampled_ppmi)
            sampled_ppmi = self._standardize_moca_suffixes(sampled_ppmi)
            sampled_ppmi["COHORT"] = "PPMI"

            # Ensure UPDRS columns (for "both" we tried to attach both; for single-state we carry whatever exists)
            for col in ("UPDRS_on", "UPDRS_off"):
                if col not in sampled_ppmi.columns:
                    sampled_ppmi[col] = sampled_ppmi.get(col, np.nan)

            return {"ppmi": sampled_ppmi, "bin_plan": bin_plan.reset_index(drop=True), "balance": bal}

        # coarsen and retry
        bins -= 1

        # If all bin sizes failed
        return {"ppmi": ppmi.iloc[0:0].copy(), "bin_plan": pd.DataFrame(), "balance": pd.DataFrame()}





    def select_medication_group(
        self,
        csv_path: str,
        quest: str,
        id_column: str,
        *,
        use_updrs: bool = True,
        updrs_state: str = "on",
        top_k: int = 8,
        bandwidth_scale: float = 1.0,
        random_state: int | None = None,
        unique_patient: bool = True,   # ensure each PPMI PATNO is used at most once
        require_ledd: bool = True,     # <— NEW: if False, do NOT require LEDD_pre on both sides
    ) -> Dict[str, pd.DataFrame]:
        """
        Build a medication (non-DBS) comparison cohort by mimicking the treated distribution
        of key covariates instead of enforcing unique 1:1 matches. Each treated participant
        draws a PPMI follow-up pair from the pool using soft nearest-neighbour weighting.

        If unique_patient=True, each PPMI PATNO is used at most once across all selections.
        If require_ledd=True, both cohorts must have non-missing LEDD_pre for inclusion.
        """
        prep = self._prepare_non_dbs_data(
            csv_path=csv_path,
            quest=quest,
            id_column=id_column,
            use_updrs=use_updrs,
            updrs_state=updrs_state,
        )
        custom_model = prep["custom_model"].copy()
        ppmi_model = prep["ppmi_model"].copy()
        updrs_col = prep["updrs_col"]

        if custom_model.empty:
            raise ValueError("Custom cohort is empty after preprocessing.")
        if ppmi_model.empty:
            raise ValueError("PPMI pool is empty after preprocessing.")

        custom = custom_model.copy()
        ppmi = ppmi_model.copy()

        # soft-match features
        dist_features: Tuple[str, ...] = ("TimeSinceSurgery", "TimeSinceDiag", "AGE_AT_OP")
        alias_map: Dict[str, Sequence[str]] = {
            "TimeSinceSurgery": ("TimeSinceSurgery",),
            "TimeSinceDiag": ("TimeSinceDiag",),
            "AGE_AT_OP": ("AGE_AT_OP", "AGE_AT_BASELINE"),
        }

        def _ensure_alias(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> None:
            for cand in candidates:
                if cand in df.columns:
                    df[target] = pd.to_numeric(df[cand], errors="coerce")
                    return
            if target not in df.columns:
                df[target] = np.nan

        for target, candidates in alias_map.items():
            _ensure_alias(custom, target, candidates)
            _ensure_alias(ppmi, target, candidates)

        for col in dist_features[:2]:
            if col in custom.columns:
                custom[col] = pd.to_numeric(custom[col], errors="coerce")
            if col in ppmi.columns:
                ppmi[col] = pd.to_numeric(ppmi[col], errors="coerce")
        if "AGE_AT_OP" in custom.columns:
            custom["AGE_AT_OP"] = pd.to_numeric(custom["AGE_AT_OP"], errors="coerce")
        if "AGE_AT_OP" in ppmi.columns:
            ppmi["AGE_AT_OP"] = pd.to_numeric(ppmi["AGE_AT_OP"], errors="coerce")

        # --- LEDD requirement (optional) ---
        if require_ledd:
            if "LEDD_pre" not in custom.columns or "LEDD_pre" not in ppmi.columns:
                raise ValueError("LEDD_pre missing. Set require_ledd=False to skip this requirement.")
            custom = custom[custom["LEDD_pre"].notna()].copy()
            ppmi = ppmi[ppmi["LEDD_pre"].notna()].copy()
            if custom.empty or ppmi.empty:
                raise ValueError("After enforcing LEDD_pre availability, one cohort is empty.")
        else:
            # Ensure column exists for downstream keep-columns (can be NaN)
            if "LEDD_pre" not in custom.columns:
                custom["LEDD_pre"] = np.nan
            if "LEDD_pre" not in ppmi.columns:
                ppmi["LEDD_pre"] = np.nan
            # Do NOT filter rows by LEDD_pre

        # Exclude held-out PATNOs (test set)
        exclude_patnos: Set[str] = set()
        for candidate in ("eligible_ppmi.csv", "eligible_matched_ppmi.csv"):
            p = Path(candidate)
            if not p.exists():
                continue
            try:
                heldout_df = pd.read_csv(p)
            except Exception as exc:
                print(f"[WARN] Could not read {candidate}: {exc}")
                continue
            for col in ("PATNO", "ppmi_patno", "matched_ppmi_patno"):
                if col in heldout_df.columns:
                    exclude_patnos.update(heldout_df[col].dropna().astype(str).tolist())
                    break
        if exclude_patnos:
            ppmi = ppmi[~ppmi["PATNO"].astype(str).isin(exclude_patnos)].copy()
            if ppmi.empty:
                raise ValueError("All PPMI candidates excluded by eligible list; nothing left to sample.")

        # feature matrix (impute pooled medians)
        feature_cols = [
            col for col in dist_features
            if col in custom.columns
            and col in ppmi.columns
            and pd.to_numeric(custom[col], errors="coerce").notna().any()
            and pd.to_numeric(ppmi[col], errors="coerce").notna().any()
        ]
        if not feature_cols:
            raise ValueError("None of the requested covariates are available on both cohorts.")

        custom_feat = custom[feature_cols].apply(pd.to_numeric, errors="coerce")
        ppmi_feat = ppmi[feature_cols].apply(pd.to_numeric, errors="coerce")
        combined_feat = pd.concat([custom_feat, ppmi_feat], axis=0, ignore_index=True)
        medians = combined_feat.median()
        valid_cols = [col for col in feature_cols if not pd.isna(medians.get(col))]
        if not valid_cols:
            raise ValueError("No usable covariates after filtering columns with all missing values.")
        fill_map = medians[valid_cols].to_dict()
        custom_feat = custom_feat[valid_cols].fillna(fill_map)
        ppmi_feat = ppmi_feat[valid_cols].fillna(fill_map)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(pd.concat([custom_feat, ppmi_feat], axis=0))
        custom_scaled = scaler.transform(custom_feat)
        ppmi_scaled = scaler.transform(ppmi_feat)

        rng = np.random.default_rng(random_state)
        top_k = max(1, min(int(top_k), len(ppmi)))  # use actual ppmi length
        ppmi_index_array = ppmi.index.to_numpy()
        if bandwidth_scale <= 0:
            raise ValueError("bandwidth_scale must be positive.")

        # --- Precompute neighbor lists & weights for each treated row ---
        custom_indices: List[int] = list(custom.index)
        neighbor_lists: List[np.ndarray] = []
        neighbor_weights: List[np.ndarray] = []
        neighbor_dists: List[np.ndarray] = []
        full_sorted_lists: List[np.ndarray] = []  # for fallback when all top-k are taken

        for order, ci in enumerate(custom_indices):
            vec = custom_scaled[order]
            dists = np.linalg.norm(ppmi_scaled - vec, axis=1)
            if not np.isfinite(dists).any():
                raise ValueError("Distance computation produced non-finite values.")

            sorted_idx = np.argsort(dists)
            full_sorted_lists.append(sorted_idx)

            k = min(top_k, len(sorted_idx))
            top_idx = sorted_idx[:k]
            top_dists = dists[top_idx]

            positive = top_dists[top_dists > 1e-12]
            if positive.size:
                base_bandwidth = float(np.median(positive))
            else:
                base_bandwidth = float(np.mean(top_dists)) if top_dists.size else 1.0
            bandwidth = max(base_bandwidth * float(bandwidth_scale), 1e-6)

            weights = np.exp(-(top_dists ** 2) / (2 * bandwidth ** 2))
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            probs = (weights / weights.sum()) if weights.sum() > 0 else np.full(len(top_idx), 1.0 / len(top_idx))

            neighbor_lists.append(top_idx)
            neighbor_weights.append(probs)
            neighbor_dists.append(top_dists)

        # --- Assign controls with unique PATNO constraint (greedy) ---
        used_patnos: Set[str] = set()
        selected_positions: List[int] = []
        pair_rows: List[Dict[str, object]] = []

        treated_order = list(range(len(custom_indices)))
        rng.shuffle(treated_order)

        for ord_pos in treated_order:
            ci = custom_indices[ord_pos]
            top_idx = neighbor_lists[ord_pos]
            probs = neighbor_weights[ord_pos]
            d_top = neighbor_dists[ord_pos]

            # Draw a ranked preference list by sampling without replacement according to probs
            if len(top_idx) == 1:
                ranked = np.array([0], dtype=int)
            else:
                ranked = rng.choice(len(top_idx), size=len(top_idx), replace=False, p=probs)

            chosen_local_idx = None
            chosen_global_pos = None
            chosen_weight = None
            chosen_dist = None

            # try within top-k, skipping already used PATNOs
            for local_idx in ranked:
                pos = int(top_idx[local_idx])
                patno = str(ppmi.iloc[pos]["PATNO"])
                if (not unique_patient) or (patno not in used_patnos):
                    chosen_local_idx = int(local_idx)
                    chosen_global_pos = pos
                    chosen_weight = float(probs[local_idx])
                    chosen_dist = float(d_top[local_idx])
                    break

            # fallback: scan beyond top-k for first unused PATNO
            if chosen_global_pos is None:
                sorted_all = full_sorted_lists[ord_pos]
                dists_all = np.linalg.norm(ppmi_scaled - custom_scaled[ord_pos], axis=1)
                if len(d_top) > 0:
                    positive = d_top[d_top > 1e-12]
                    base_bandwidth = float(np.median(positive)) if positive.size else float(np.mean(d_top)) if d_top.size else 1.0
                else:
                    base_bandwidth = 1.0
                bandwidth = max(base_bandwidth * float(bandwidth_scale), 1e-6)

                for pos in sorted_all[top_k:]:
                    patno = str(ppmi.iloc[int(pos)]["PATNO"])
                    if (not unique_patient) or (patno not in used_patnos):
                        chosen_global_pos = int(pos)
                        chosen_dist = float(dists_all[int(pos)])
                        chosen_weight = float(np.exp(-(chosen_dist ** 2) / (2 * bandwidth ** 2)))
                        chosen_local_idx = None  # indicates fallback
                        break

            if chosen_global_pos is None:
                # last resort: reuse if uniqueness makes it impossible
                if unique_patient:
                    pos = int(full_sorted_lists[ord_pos][0])
                    patno = str(ppmi.iloc[pos]["PATNO"])
                    chosen_global_pos = pos
                    chosen_dist = float(np.linalg.norm(ppmi_scaled[pos] - custom_scaled[ord_pos]))
                    chosen_weight = 0.0
                    chosen_local_idx = None
                else:
                    best = int(np.argmax(probs))
                    chosen_global_pos = int(top_idx[best])
                    chosen_dist = float(d_top[best])
                    chosen_weight = float(probs[best])
                    chosen_local_idx = best

            pos = int(chosen_global_pos)
            patno = str(ppmi.iloc[pos]["PATNO"])
            if unique_patient:
                used_patnos.add(patno)

            selected_positions.append(pos)
            pair_rows.append({
                "custom_index": int(ci),
                "custom_id": str(custom_model.at[ci, id_column]),
                "ppmi_index": int(ppmi_index_array[pos]),
                "ppmi_patno": patno,
                "distance": float(chosen_dist),
                "neighbor_rank": int((0 if chosen_local_idx is None else chosen_local_idx) + 1),
                "weight_used": float(chosen_weight),
                "fallback_used": bool(chosen_local_idx is None and pos not in neighbor_lists[ord_pos]),
            })

        # outputs
        selected_custom = custom.loc[custom_indices].copy()
        selected_ppmi = ppmi.iloc[selected_positions].copy()

        selected_custom["matched_ppmi_patno"] = selected_ppmi["PATNO"].values
        selected_custom["matched_ppmi_index"] = selected_ppmi.index.values
        selected_ppmi["matched_custom_id"] = selected_custom[id_column].values
        selected_ppmi["matched_custom_index"] = selected_custom.index.values

        # normalize MoCA naming
        selected_custom = self._rename_moca_pre_post(selected_custom)
        selected_ppmi = self._rename_moca_pre_post(selected_ppmi)
        selected_custom = self._standardize_moca_suffixes(selected_custom)
        selected_ppmi = self._standardize_moca_suffixes(selected_ppmi)

        # balance diagnostics (SMD)
        def _smd(a: pd.Series, b: pd.Series) -> float:
            a = pd.to_numeric(a, errors="coerce").dropna()
            b = pd.to_numeric(b, errors="coerce").dropna()
            if len(a) < 2 or len(b) < 2:
                return np.nan
            m1, m0 = a.mean(), b.mean()
            s1, s0 = a.std(ddof=1), b.std(ddof=1)
            denom = np.sqrt((s1**2 + s0**2) / 2.0)
            return float((m1 - m0) / denom) if denom > 0 else np.nan

        balance_rows = []
        for col in valid_cols:
            balance_rows.append({
                "covariate": col,
                "custom_mean": float(pd.to_numeric(selected_custom[col], errors="coerce").mean(skipna=True)),
                "ppmi_mean": float(pd.to_numeric(selected_ppmi[col], errors="coerce").mean(skipna=True)),
                "post_SMD": _smd(selected_custom[col], selected_ppmi[col]),
                "pre_SMD": _smd(custom[col], ppmi[col]),
            })
        balance = pd.DataFrame(balance_rows).set_index("covariate")

        pairs = pd.DataFrame(
            pair_rows,
            columns=["custom_index", "custom_id", "ppmi_index", "ppmi_patno", "distance",
                     "neighbor_rank", "weight_used", "fallback_used"],
        )

        treated_out = self._non_dbs_keep_columns(
            selected_custom, "Custom", id_column=id_column, updrs_col=updrs_col
        )
        control_out = self._non_dbs_keep_columns(
            selected_ppmi, "PPMI", id_column=id_column, updrs_col=updrs_col
        )

        return {
            "custom": treated_out,
            "ppmi": control_out,
            "pairs": pairs,
            "balance": balance,
        }






    def plot_time_since_diag_hist(self, df, cohort_name="PPMI", bins=30):
        """
        Draws histogram + kernel density of TimeSinceDiag (in years).
        """
        df = df.copy()
        col = "TimeSinceDiag"
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe.")
        df = df[pd.to_numeric(df[col], errors="coerce").notna()]
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=bins, kde=True, color="steelblue", alpha=0.7)
        plt.xlabel("Time Since Diagnosis (years)")
        plt.ylabel("Number of Visits")
        plt.title(f"{cohort_name}: Distribution of Time Since Diagnosis (N={len(df)})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{cohort_name}_TimeSinceDiag_Histogram.png")

    @staticmethod
    def iterative_propensity_matching(
        custom_model: pd.DataFrame,
        ppmi_model: pd.DataFrame,
        time_tolerance_days: float,
        caliper_logit: float,
        *,
        replace: bool = False,
        batch_size: int = 500,
        random_state: int = 42,
        # existing hard calipers
        caliper_age_years: float | None = 5.0,
        caliper_tsd_years: float | None = 2.0,
        exact_sex: bool = False,
        # NEW: hard caliper on TimeSinceSurgery
        caliper_tss_years: float | None = 0.75,      # ±9 months by default
        # weights for ranking
        w_logit: float = 1.0,
        w_time: float = 0.25,
        w_age: float = 0.75,
        w_tsd: float = 0.50,
        w_updrs: float = 0.25,
        # NEW: weight for TSS in ranking
        w_tss: float = 0.75,
        ) ->     tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Iterative matching using *precomputed* 'logit' and hard calipers.
        Adds a hard caliper and weighted cost term for TimeSinceSurgery (years).
        Returns (treated_matches, control_matches, pairs_df).
        """
        rng = np.random.default_rng(random_state)

        if "__orig_ix" not in ppmi_model.columns:
            ppmi_model = ppmi_model.copy()
            ppmi_model["__orig_ix"] = ppmi_model.index

        available_ix = list(ppmi_model.index)

        t_sort_cols = ["TimeSinceBaselineDays"]
        t_sort_cols.append("PATNO" if "PATNO" in custom_model.columns else custom_model.columns[0])
        treated_order = custom_model.sort_values(t_sort_cols).index.tolist()

        matched_treated_ix: list[int] = []
        matched_control_orig_ix: list[int] = []
        pair_rows: list[dict] = []

        updrs_col = (
            "UPDRS_on" if "UPDRS_on" in ppmi_model.columns
            else ("UPDRS_off" if "UPDRS_off" in ppmi_model.columns else None)
        )

        for start in range(0, len(treated_order), batch_size):
            block = treated_order[start:start + batch_size]

            for ci in block:
                if not available_ix and not replace:
                    break

                trow = custom_model.loc[ci]
                pool = ppmi_model.loc[available_ix] if not replace else ppmi_model

                # ---------- HARD FILTERS ----------
                # time alignment
                cand = pool[
                    (pool["TimeSinceBaselineDays"] - trow["TimeSinceBaselineDays"]).abs()
                    <= float(time_tolerance_days)
                ].copy()
                if cand.empty:
                    continue

                # TimeSinceDiag (years)
                if caliper_tsd_years is not None and "TimeSinceDiag" in cand.columns:
                    tsd = trow.get("TimeSinceDiag", np.nan)
                    cand = cand[(cand["TimeSinceDiag"] - tsd).abs() <= float(caliper_tsd_years)]
                    if cand.empty:
                        continue

                # Age (years)
                if caliper_age_years is not None and "AGE_AT_BASELINE" in cand.columns:
                    age = trow.get("AGE_AT_BASELINE", np.nan)
                    cand = cand[(cand["AGE_AT_BASELINE"] - age).abs() <= float(caliper_age_years)]
                    if cand.empty:
                        continue

                # NEW: TimeSinceSurgery (years)
                if caliper_tss_years is not None and "TimeSinceSurgery" in cand.columns and "TimeSinceSurgery" in trow.index:
                    tss = trow.get("TimeSinceSurgery", np.nan)
                    cand = cand[(cand["TimeSinceSurgery"] - tss).abs() <= float(caliper_tss_years)]
                    if cand.empty:
                        continue

                # exact SEX (optional)
                if exact_sex and ("SEX" in cand.columns) and ("SEX" in trow.index):
                    cand = cand[cand["SEX"] == trow["SEX"]]
                    if cand.empty:
                        continue

                # logit caliper
                cand = cand[(cand["logit"] - trow["logit"]).abs() <= float(caliper_logit)].copy()
                if cand.empty:
                    continue

                # ---------- RANK BY WEIGHTED COST ----------
                cand["logit_diff"] = (cand["logit"] - trow["logit"]).abs()
                cand["time_diff"]  = (cand["TimeSinceBaselineDays"] - trow["TimeSinceBaselineDays"]).abs()

                cand["diag_diff"] = np.inf
                if "TimeSinceDiag" in cand.columns:
                    cand["diag_diff"] = np.abs(cand["TimeSinceDiag"] - trow.get("TimeSinceDiag", np.nan))

                cand["age_diff"] = np.inf
                if "AGE_AT_BASELINE" in cand.columns:
                    cand["age_diff"] = np.abs(cand["AGE_AT_BASELINE"] - trow.get("AGE_AT_BASELINE", np.nan))

                cand["updrs_diff"] = np.inf
                if updrs_col is not None and updrs_col in cand.columns:
                    cand["updrs_diff"] = np.abs(cand[updrs_col] - trow.get(updrs_col, np.nan))

                # NEW: TSS diff
                cand["tss_diff"] = np.inf
                if "TimeSinceSurgery" in cand.columns and "TimeSinceSurgery" in trow.index:
                    cand["tss_diff"] = np.abs(cand["TimeSinceSurgery"] - trow.get("TimeSinceSurgery", np.nan))

                eps = 1e-9
                cand["time_z"]  = cand["time_diff"]  / (float(time_tolerance_days) + eps)
                cand["age_z"]   = cand["age_diff"]   / (float(caliper_age_years) + eps if caliper_age_years else (cand["age_diff"].max() + 1.0))
                cand["tsd_z"]   = cand["diag_diff"]  / (float(caliper_tsd_years) + eps if caliper_tsd_years else (cand["diag_diff"].max() + 1.0))
                cand["tss_z"]   = cand["tss_diff"]   / (float(caliper_tss_years) + eps if caliper_tss_years else (cand["tss_diff"].max() + 1.0))
                cand["updrs_z"] = cand["updrs_diff"] / (cand["updrs_diff"].max() + 1.0)

                cand["cost"] = (
                    w_logit * cand["logit_diff"]
                    + w_time  * cand["time_z"]
                    + w_age   * cand["age_z"]
                    + w_tsd   * cand["tsd_z"]
                    + w_tss   * cand["tss_z"]        # NEW
                    + w_updrs * cand["updrs_z"]
                )

                chosen_pi = int(cand.sort_values(["cost", "PATNO", "TEST_DATUM"]).index[0])
                chosen = cand.loc[chosen_pi]

                matched_treated_ix.append(ci)
                matched_control_orig_ix.append(int(chosen["__orig_ix"]))
                pair_rows.append({
                    "custom_index": int(ci),
                    "ppmi_index": int(chosen_pi),
                    "ppmi_orig_index": int(chosen["__orig_ix"]),
                    "time_diff_days": float(chosen["time_diff"]),
                    "logit_diff": float(chosen["logit_diff"]),
                    "age_diff": float(chosen.get("age_diff", np.nan)),
                    "tsd_diff": float(chosen.get("diag_diff", np.nan)),
                    "tss_diff": float(chosen.get("tss_diff", np.nan)),  # NEW
                })

                if not replace and chosen_pi in available_ix:
                    available_ix.remove(chosen_pi)

        treated_matches = custom_model.loc[matched_treated_ix].copy().reset_index(drop=True)
        control_matches = (
            ppmi_model.set_index("__orig_ix")
            .loc[matched_control_orig_ix]
            .copy()
            .reset_index(drop=True)
        )
        pairs = pd.DataFrame(pair_rows)

        return treated_matches, control_matches, pairs



    # ---------- Matching: DBS ----------
    def match_dbs(
        self,
        csv_path: str,
        quest: str,
        STN: bool = True,
        use_updrs: bool = True,
        updrs_state: str = "on",  # 'off' or 'on'
    ) -> pd.DataFrame:
        # Load external outcome file
        df = pd.read_csv(csv_path)
        df = df.rename(columns={c: c.replace('_pre', '') for c in df.columns if c.endswith('_pre')})
        df = df[[c for c in df.columns if not c.endswith('_post')]]

        # Load PPMI raw + convert (DBS=True)
        raw = self.load_ppmi()
        ppmi_cd = self.convert_to_standard_keys(raw, DBS=True)

        # Merge demographic + questionnaire
        if quest.lower() != "moca":
            raise NotImplementedError("Only quest='moca' currently supported.")
        moca_with_demo = self._prepare_moca_with_demo(ppmi_cd)
        if moca_with_demo.empty:
            return moca_with_demo

        essential_cols = [
            "PATNO","OP_DATUM","TEST_DATUM","MoCA_sum","LOCATION","DIAG_DATE","SEX","AGE_AT_OP","LEDD_pre"
        ]
        available_cols = [c for c in essential_cols if c in moca_with_demo.columns]
        pc_df = moca_with_demo[available_cols].copy()

        # add category sums if present
        for agg_col in self.MOCA_CATEGORY_MAP:
            for variant in (agg_col, f"{agg_col}_sum"):
                if variant in moca_with_demo.columns:
                    pc_df[variant] = moca_with_demo[variant]

        subset_cols = [c for c in ["PATNO", "OP_DATUM", "TEST_DATUM", "LOCATION"] if c in pc_df.columns]
        pc_df_before_merge = pc_df[subset_cols].copy()

        # Time metrics
        pc_df = safe_parse_dates(pc_df, cols=["TEST_DATUM", "OP_DATUM", "DIAG_DATE"], dayfirst=True, report=True)
        pc_df_before_merge = safe_parse_dates(pc_df_before_merge, cols=["TEST_DATUM","OP_DATUM"], dayfirst=True, report=True)

        pc_df["TimeSinceSurgery"] = (pc_df["TEST_DATUM"] - pc_df["OP_DATUM"]).dt.days / 365.25
        pc_df_before_merge["TimeSinceSurgery"] = (pc_df_before_merge["TEST_DATUM"] - pc_df_before_merge["OP_DATUM"]).dt.days / 365.25
        pc_df_fu = pc_df[pc_df["TimeSinceSurgery"] > 0].copy()
        pc_df["TimeSinceDiag"] = (pc_df["TEST_DATUM"] - pc_df["DIAG_DATE"]).dt.days / 365.25

        if STN and "LOCATION" in pc_df.columns:
            pc_df = pc_df[pc_df["LOCATION"] == "STN"].drop(columns=["LOCATION"])
            pc_df_fu = pc_df_fu[pc_df_fu["LOCATION"] == "STN"]

        # ---------- Attach UPDRS III by chosen state ----------
        if use_updrs:
            updrs_df = ppmi_cd["mds_updrs"]["mds_updrs3"].copy()
            state = updrs_state.lower().strip()
            off_on_df = self._get_updrs3_by_state(updrs_df, state=state)  # PATNO, TEST_DATUM, UPDRS_off|UPDRS_on

            merged = pd.merge(
                pc_df,
                off_on_df.rename(columns={"TEST_DATUM": "UPDRS_TEST_DATUM"}),
                on="PATNO",
                how="inner"
            )
            merged["UPDRSET_dist"] = (merged["UPDRS_TEST_DATUM"] - merged["TEST_DATUM"]).dt.days.abs()
            merged = merged[(merged["UPDRSET_dist"] <= 730) & (merged["UPDRS_TEST_DATUM"] < merged["OP_DATUM"])]

            pc_df = (
                merged.sort_values("UPDRSET_dist")
                .groupby(["PATNO", "TEST_DATUM"], as_index=False)
                .first()
                .dropna(subset=[f"UPDRS_{state}", "UPDRS_TEST_DATUM"])
            )
            pc_df = pc_df.copy()

        # baseline & follow-up
        pc_df_bl = pc_df[pc_df["TimeSinceSurgery"] <= 0].copy()
        pc_df_bl = pc_df_bl.drop(columns=["TimeSinceSurgery","TEST_DATUM","DIAG_DATE","UPDRS_TEST_DATUM","UPDRSET_dist"], errors="ignore")
        drop_fu_cols = ["TimeSinceDiag","AGE_AT_OP","OP_DATUM","SEX","DIAG_DATE","UPDRS_TEST_DATUM","UPDRSET_dist","LEDD_pre"]
        pc_df_fu = pc_df_fu.drop(columns=drop_fu_cols, errors="ignore")

        if use_updrs:
            updrs_value_cols = [c for c in pc_df_bl.columns if c.startswith("UPDRS_")]
            if updrs_value_cols:
                pc_df_bl = pc_df_bl.rename(columns={col: f"{col}_pre" for col in updrs_value_cols})

        suffix_cols_bl = [c for c in pc_df_bl.columns if c.startswith("MoCA_")]
        suffix_cols_fu = [c for c in pc_df_fu.columns if c.startswith("MoCA_")]
        pc_df_bl = pc_df_bl.rename(columns={c: f"{c}_pre" for c in suffix_cols_bl})
        pc_df_fu = pc_df_fu.rename(columns={c: f"{c}_post" for c in suffix_cols_fu})

        combined = pd.merge(pc_df_bl, pc_df_fu, on=["PATNO"], how="inner")
        combined["TimeSinceSurgery_abs_diff"] = (combined["TimeSinceSurgery"] - 2).abs()
        closest_followups = combined.loc[combined.groupby("PATNO")["TimeSinceSurgery_abs_diff"].idxmin()].reset_index(drop=True)
        final_df = closest_followups.drop(columns=["TimeSinceSurgery_abs_diff"])

        invalid_moca = final_df["MoCA_sum_post"] > 30
        if invalid_moca.any():
            print(f"[match_dbs] Dropping {invalid_moca.sum()} rows with MoCA_sum_post > 30")
            final_df = final_df.loc[~invalid_moca].reset_index(drop=True)
        print(len(final_df))
        return final_df


    @staticmethod
    def _trim_common_support(a: pd.Series, b: pd.Series, lo_q=0.05, hi_q=0.95) -> tuple[float, float]:
        """
        Return [lo, hi] that defines the intersection of the central quantile bands
        of two distributions. Caller decides how to apply it to dataframes.
        """
        a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        a, b = a.dropna(), b.dropna()
        if a.empty or b.empty:
            return -np.inf, np.inf
        lo = max(a.quantile(lo_q), b.quantile(lo_q))
        hi = min(a.quantile(hi_q), b.quantile(hi_q))
        if lo > hi:  # no overlap — return degenerate band to filter all
            return np.inf, -np.inf
        return float(lo), float(hi)
    
    
    @staticmethod
    def _mahalanobis(X: pd.DataFrame, x_row: pd.Series) -> pd.Series:
        """
        Compute Mahalanobis distance from every row in X to x_row.
        X must contain only numeric columns. Falls back to standardized
        Euclidean if covariance is singular.
        """
        Z = X.astype(float).values
        z = x_row.astype(float).values
        diff = Z - z  # (n, p)
        # Covariance + inverse
        cov = np.cov(Z, rowvar=False)
        try:
            VI = np.linalg.pinv(cov)  # robust inverse
        except Exception:
            VI = np.eye(cov.shape[0])
        d = np.sqrt(np.sum(diff @ VI * diff, axis=1))
        return pd.Series(d, index=X.index)


        # ---------- Matching: non-DBS ----------
    def match_non_dbs(
            self,
            csv_path: str,
            quest: str,
            id_column: str,
            time_tolerance_days: int = 240,
            use_updrs: bool = True,
            updrs_state: str = "on",     # 'off' or 'on'
            replace: bool = False,
            random_state: int | None = None,
            tsd_caliper_years: float = 2.0,
            logit_caliper_sd: float = np.inf,  # if == -1 -> use iterative selector path
        ) -> Dict[str, pd.DataFrame]:
        """
        Safe non-DBS matching:
        - Propensity uses ONLY pre-treatment covariates (UPDRS_{state}, MoCA_sum_pre if present,
          and optionally AGE_AT_BASELINE).
        - Hard caliper on visit-time alignment (± time_tolerance_days).
        - Optional logit-propensity caliper (± logit_caliper_sd * SD(logit)).
        - Deterministic greedy 1:1 matching (or iterative selector when flag set).
        - Returns matched rows + pair diagnostics + SMD balance table + imputation summary.
        """

        prep = self._prepare_non_dbs_data(
            csv_path=csv_path,
            quest=quest,
            id_column=id_column,
            use_updrs=use_updrs,
            updrs_state=updrs_state,
        )
        custom_df = prep["custom_df"]
        custom_model = prep["custom_model"]
        ppmi_model = prep["ppmi_model"]
        updrs_col = prep["updrs_col"]
        state = prep["updrs_state"]

        # ---- Covariates (pre-treatment) ----
        covars = ["TimeSinceDiag"]
        if use_updrs and updrs_col in ppmi_model.columns:
            covars.append(updrs_col)
        moca_covars = ["MoCA_sum_pre", "MoCA_Erinnerung_sum_pre", "MoCA_Executive_sum_pre"]
        covars.extend([c for c in moca_covars if c in ppmi_model.columns])
        if "AGE_AT_BASELINE" in ppmi_model.columns:
            covars.append("AGE_AT_BASELINE")

        # Ensure covariates exist in custom_df (placeholder if missing)
        for c in covars:
            if c not in custom_df.columns:
                custom_df[c] = np.nan

        # ---- STRICT: require TimeSinceDiag on both sides (no imputation) ----
        custom_model = custom_model.dropna(subset=["TimeSinceDiag"]).copy()
        ppmi_model   = ppmi_model.dropna(subset=["TimeSinceDiag"]).copy()
        if custom_model.empty or ppmi_model.empty:
            raise ValueError("After enforcing non-missing TimeSinceDiag, one cohort is empty.")

        # ---- Propensity pipeline: impute (median) -> scale -> logit (fit ONCE globally)
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        prop_model = Pipeline([
            ("prep", ColumnTransformer([
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())
                ]), covars)
            ], remainder="drop", verbose_feature_names_out=False)),
            ("logit", LogisticRegression(
                solver="liblinear",
                C=1.0,
                max_iter=2000,
                class_weight="balanced"
            ))
        ])

        # Design matrices BEFORE fitting
        X_custom = custom_model[covars]
        X_ppmi   = ppmi_model[covars]
        X_all    = pd.concat([X_custom, X_ppmi], axis=0, ignore_index=True)
        y_all    = np.concatenate([np.ones(len(X_custom), dtype=int),
                                   np.zeros(len(X_ppmi), dtype=int)])

        print("[match_non_dbs] Matching covariates used in propensity model:")
        for c in covars:
            n_custom = custom_model[c].notna().sum()
            n_ppmi = ppmi_model[c].notna().sum()
            print(f"  {c:30s} | custom: {n_custom:4d} non-NA | ppmi: {n_ppmi:4d} non-NA")

        print(f"[DEBUG] Final covariates used in matching: {covars}")
        prop_model.fit(X_all, y_all)

        # ---- Imputation summary (counts from pre-imputation matrices)
        imputations_summary = []
        for col in covars:
            n_miss_custom = X_custom[col].isna().sum()
            n_miss_ppmi   = X_ppmi[col].isna().sum()
            n_miss_total  = int(n_miss_custom + n_miss_ppmi)
            imputations_summary.append({
                "column": col,
                "n_missing_custom": int(n_miss_custom),
                "n_missing_ppmi": int(n_miss_ppmi),
                "n_missing_total": n_miss_total,
                "prop_missing_total": float(n_miss_total / len(X_all))
            })
        num_pipe = prop_model.named_steps["prep"].named_transformers_["num"]
        medians = getattr(num_pipe.named_steps["imp"], "statistics_", None)
        if medians is not None:
            for i, col in enumerate(covars):
                imputations_summary[i]["imputed_median"] = float(medians[i])
        imputations_summary = pd.DataFrame(imputations_summary)
        print("[match_non_dbs] Imputation summary:")
        print(imputations_summary.to_string(index=False))

        # Predicted propensity and logits (GLOBAL)
        p_custom = prop_model.predict_proba(X_custom)[:, 1]
        p_ppmi   = prop_model.predict_proba(X_ppmi)[:, 1]

        eps = 1e-6
        l_custom = np.log(np.clip(p_custom, eps, 1 - eps) / np.clip(1 - p_custom, eps, 1 - eps))
        l_ppmi   = np.log(np.clip(p_ppmi,   eps, 1 - eps) / np.clip(1 - p_ppmi,   eps, 1 - eps))
        sd_logit = np.std(np.concatenate([l_custom, l_ppmi]))
        caliper  = (logit_caliper_sd * sd_logit) if sd_logit > 0 else np.inf

        custom_model = custom_model.assign(propensity=p_custom, logit=l_custom)
        ppmi_model   = ppmi_model.assign(propensity=p_ppmi,   logit=l_ppmi)

        # ---- Path 1: Iterative selector (no re-training) ----
        if logit_caliper_sd == -1:
            print("[match_non_dbs] Using iterative selection with global propensities")
            if "__orig_ix" not in ppmi_model.columns:
                ppmi_model["__orig_ix"] = ppmi_model.index

            treated_matches, control_matches, pair_df = self.iterative_propensity_matching(
                custom_model=custom_model,
                ppmi_model=ppmi_model,
                time_tolerance_days=time_tolerance_days,
                caliper_logit=sd_logit * 0.2 if np.isfinite(sd_logit) and sd_logit > 0 else np.inf,
                replace=replace,
                batch_size=500,
                random_state=42,
            )
            treated_matches["matched_ppmi_patno"]  = control_matches["PATNO"].values
            treated_matches["matched_ppmi_index"]  = pair_df["ppmi_orig_index"].values
            control_matches["matched_custom_index"] = pair_df["custom_index"].values
            control_matches["matched_custom_id"]    = treated_matches[id_column].values
            treated_matches = self._rename_moca_pre_post(treated_matches)
            control_matches = self._rename_moca_pre_post(control_matches)
            treated_matches = self._standardize_moca_suffixes(treated_matches)
            control_matches = self._standardize_moca_suffixes(control_matches)

            treated_matches = self._non_dbs_keep_columns(
                treated_matches, "Custom", id_column=id_column, updrs_col=updrs_col
            )
            control_matches = self._non_dbs_keep_columns(
                control_matches, "PPMI", id_column=id_column, updrs_col=updrs_col
            )

            return {
                "custom": treated_matches,
                "ppmi": control_matches,
                "pairs": pair_df,
                "balance": pd.DataFrame(),
                "imputation_summary": imputations_summary
            }

        # ---- Path 2: Deterministic greedy matching with hard calipers ----
        pool_idx = list(ppmi_model.index)
        matched_ci, matched_pi, records = [], [], []

        custom_order = custom_model.sort_values(["TimeSinceBaselineDays", id_column]).index.tolist()

        for ci in custom_order:
            if not pool_idx and not replace:
                break

            row = custom_model.loc[ci]
            pool = ppmi_model.loc[pool_idx] if not replace else ppmi_model

            # 1) time caliper (hard)
            cand = pool[(pool["TimeSinceBaselineDays"] - row["TimeSinceBaselineDays"]).abs() <= float(time_tolerance_days)].copy()
            if cand.empty:
                continue

            # 2) optional TSD proximity caliper
            if "TimeSinceDiag" in cand.columns and "TimeSinceDiag" in row:
                cand = cand[(cand["TimeSinceDiag"] - row["TimeSinceDiag"]).abs() <= tsd_caliper_years]
                if cand.empty:
                    continue

            # 3) logit caliper (hard)
            cand = cand[(cand["logit"] - row["logit"]).abs() <= caliper].copy()
            if cand.empty:
                continue

            # Rank: propensity diff, time diff, |Δ TimeSinceDiag|, |Δ UPDRS|
            cand["prop_diff"] = (cand["propensity"] - row["propensity"]).abs()
            cand["time_diff"] = (cand["TimeSinceBaselineDays"] - row["TimeSinceBaselineDays"]).abs()

            if "TimeSinceDiag" in cand.columns:
                diag_diff = np.abs(cand["TimeSinceDiag"] - row.get("TimeSinceDiag", np.nan))
                cand["diag_diff"] = np.nan_to_num(diag_diff, nan=np.inf)
            else:
                cand["diag_diff"] = np.inf

            if updrs_col in cand.columns:
                u_diff = np.abs(cand[updrs_col] - row.get(updrs_col, np.nan))
                cand["updrs_diff"] = np.nan_to_num(u_diff, nan=np.inf)
            else:
                cand["updrs_diff"] = np.inf

            cand["age_diff"] = np.abs(cand["AGE_AT_BASELINE"] - row["AGE_AT_BASELINE"])
            cand = cand.sort_values(["prop_diff", "diag_diff","age_diff", "time_diff", "updrs_diff",  "PATNO", "TEST_DATUM"])
            pi = int(cand.index[0])
            pr = cand.loc[pi]

            matched_ci.append(ci)
            matched_pi.append(pi)
            records.append({
                "custom_index": int(ci),
                "custom_id":   str(row[id_column]),
                "ppmi_index":  int(pi),
                "ppmi_patno":  pr["PATNO"],
                "time_diff_days": float(pr["time_diff"]),
                "propensity_diff": float(pr["prop_diff"]),
                "logit_diff": float(abs(pr["logit"] - row["logit"])),
            })

            if not replace and pi in pool_idx:
                pool_idx.remove(pi)

        if not matched_ci:
            raise ValueError("No matches could be constructed under the current settings/calipers.")

        # ---- Build matched outputs
        matched_custom = custom_model.loc[matched_ci].copy()
        matched_ppmi   = ppmi_model.loc[matched_pi].copy()

        # Pairs BEFORE filtering
        pairs = pd.DataFrame.from_records(records)
        pairs = pairs.merge(
            custom_model[["propensity", "logit"]].rename(
                columns={"propensity": "custom_propensity", "logit": "custom_logit"}
            ),
            left_on="custom_index", right_index=True, how="left"
        ).merge(
            ppmi_model[["PATNO", "propensity", "logit"]].rename(
                columns={"PATNO": "ppmi_patno", "propensity": "ppmi_propensity", "logit": "ppmi_logit"}
            ),
            left_on="ppmi_index", right_index=True, how="left"
        )

        # Cross refs
        matched_custom["matched_ppmi_patno"] = matched_ppmi["PATNO"].values
        matched_custom["matched_ppmi_index"] = matched_ppmi.index.values
        matched_ppmi["matched_custom_id"]    = matched_custom[id_column].values
        matched_ppmi["matched_custom_index"] = matched_custom.index.values

        # Rename MoCA pre/post + normalize suffixes
        matched_custom = self._rename_moca_pre_post(matched_custom)
        matched_ppmi   = self._rename_moca_pre_post(matched_ppmi)
        matched_custom = self._standardize_moca_suffixes(matched_custom)
        matched_ppmi   = self._standardize_moca_suffixes(matched_ppmi)

        # Keep only needed columns (+ diagnostics)
        matched_custom = self._non_dbs_keep_columns(
            matched_custom, "Custom", id_column=id_column, updrs_col=updrs_col
        )
        matched_ppmi = self._non_dbs_keep_columns(
            matched_ppmi, "PPMI", id_column=id_column, updrs_col=updrs_col
        )

        # ---- Simple balance diagnostics: SMDs pre vs post
        def _smd(a: pd.Series, b: pd.Series) -> float:
            a = pd.to_numeric(a, errors="coerce"); a = a[~a.isna()]
            b = pd.to_numeric(b, errors="coerce"); b = b[~b.isna()]
            if len(a) < 2 or len(b) < 2:
                return np.nan
            m1, m0 = a.mean(), b.mean()
            s1, s0 = a.std(ddof=1), b.std(ddof=1)
            denom = np.sqrt((s1**2 + s0**2) / 2.0)
            return float((m1 - m0) / denom) if denom > 0 else np.nan

        covs_for_smd = ["TimeSinceDiag"] if "TimeSinceDiag" in ppmi_model.columns else []
        if updrs_col in ppmi_model.columns:
            covs_for_smd.append(updrs_col)
        if "LEDD_pre" in ppmi_model.columns:
            covs_for_smd.append("LEDD_pre")
        if "AGE_AT_BASELINE" in ppmi_model.columns:
            covs_for_smd.append("AGE_AT_BASELINE")

        pre_t = custom_model[covs_for_smd] if covs_for_smd else pd.DataFrame(index=custom_model.index)
        pre_c = ppmi_model[covs_for_smd]   if covs_for_smd else pd.DataFrame(index=ppmi_model.index)
        pre_smd  = {k: _smd(pre_t[k], pre_c[k]) for k in covs_for_smd} if covs_for_smd else {}
        post_smd = {k: _smd(matched_custom[k], matched_ppmi[k]) for k in covs_for_smd} if covs_for_smd else {}
        smd_table = pd.DataFrame({"pre_SMD": pre_smd, "post_SMD": post_smd})

        return {
            "custom": matched_custom,
            "ppmi": matched_ppmi,
            "pairs": pairs,
            "balance": smd_table,
            "imputation_summary": imputations_summary,
        }




    # ---------- Stubs (left as-is) ----------
    def load_costum(self, path_of_folder: str, foldertype: str):
        pass

    def export_covariate_names(self, full_df, path: str):
        if self.foldertype == "tuebingen":
            with open(path + '/covariate_names.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(full_df.keys())
                def write_dict_to_csv(d, parent_key=''):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            write_dict_to_csv(value)
                        else:
                            writer.writerow([parent_key + key] + list(value.columns))
                write_dict_to_csv(full_df)

    def select_covariates(self, covariates: Dict = None):
        # no persistent complete_data; keep as a stub or refactor out
        return covariates

    def convert_costom_to_standard_keys(self, file: str):
        pass

    def remove_outliers(self):
        print("Removing outliers...")
        pass


    def save_eligible_under_calipers(
        self,
        custom_csv: str,
        control_csv: str,
        *,
        # calipers
        time_tolerance_days: int = 120,
        use_logit_caliper: bool = True,
        caliper_multiplier: float = 0.2,   # 0.2 × SD(logit) = classic rule
        # schema hints (auto-detected if possible)
        custom_id_col: str = "PATNO",
        control_id_col: str = "PATNO",
        propensity_col: str | None = "propensity",   # if None, try to infer
        logit_col: str | None = None,                # if present, will be used directly
        time_col: str | None = "TimeSinceBaselineDays",  # if None, derive from dates
        date_candidates: tuple[str, ...] = ("TEST_DATUM", "EXAMDT", "INFODT"),
        # output
        out_dir: str | None = None,   # default: same dir as each input file
    ) -> dict:
        """
        Create *eligibility filters* (not full matching):
        - A treated row is kept if it has >=1 control within both calipers.
        - A control row is kept if it is eligible for >=1 treated row.
        Saves: eligible_<basename(custom_csv)>, eligible_<basename(control_csv)>.
        Returns counts & file paths.
        """

        # ---------- helpers ----------
        def _first_present(df: pd.DataFrame, cols: tuple[str, ...]) -> str | None:
            for c in cols:
                if c in df.columns:
                    return c
            return None

        def _ensure_time_since_baseline(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
            """Ensure numeric TimeSinceBaselineDays. If absent, build from the first available date col."""
            nonlocal time_col
            if time_col and time_col in df.columns:
                return df

            # pick a date column
            dcol = _first_present(df, date_candidates)
            if dcol is None:
                raise ValueError("No time variable found: either provide `time_col` or one of "
                                 f"{date_candidates} must be present to compute TimeSinceBaselineDays.")

            # parse and compute per-id baseline
            tmp = df.copy()
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
            if tmp[dcol].isna().all():
                raise ValueError(f"Could not parse any dates in column {dcol} to compute TimeSinceBaselineDays.")
            tmp = tmp.dropna(subset=[dcol])
            first_dt = tmp.groupby(id_col)[dcol].transform("min")
            tmp["TimeSinceBaselineDays"] = (tmp[dcol] - first_dt).dt.days
            return tmp

        def _ensure_logit(df: pd.DataFrame) -> pd.DataFrame:
            """Ensure 'logit' column exists, using either existing logit_col or a propensity column."""
            nonlocal propensity_col, logit_col
            out = df.copy()
            if logit_col and logit_col in out.columns:
                out["logit"] = pd.to_numeric(out[logit_col], errors="coerce")
                return out

            # find propensity
            if propensity_col is None or propensity_col not in out.columns:
                # try to infer a reasonable name if not present
                guesses = ("propensity", "pscore", "p_score", "p")
                propensity_col = _first_present(out, guesses)
                if propensity_col is None:
                    raise ValueError("No propensity probability found. Provide `propensity_col` or include one of "
                                     f"{guesses} in the CSV.")

            p = pd.to_numeric(out[propensity_col], errors="coerce").clip(0, 1)
            eps = 1e-6
            out["logit"] = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
            return out

        def _load_and_prepare(path: str, id_col: str) -> pd.DataFrame:
            df = pd.read_csv(path)
            if id_col not in df.columns:
                raise ValueError(f"{os.path.basename(path)} is missing id column '{id_col}'")
            df = _ensure_time_since_baseline(df, id_col)
            df = _ensure_logit(df)
            # Keep only rows that have both measures
            df = df.dropna(subset=["TimeSinceBaselineDays", "logit"]).copy()
            return df

        # ---------- load & prepare ----------
        T = _load_and_prepare(custom_csv, custom_id_col)
        C = _load_and_prepare(control_csv, control_id_col)

        # ---------- choose logit caliper ----------
        if use_logit_caliper:
            sd_logit = float(np.std(np.concatenate([T["logit"].values, C["logit"].values])))
            caliper = caliper_multiplier * sd_logit if sd_logit > 0 else np.inf
        else:
            sd_logit = float(np.std(np.concatenate([T["logit"].values, C["logit"].values])))
            caliper = np.inf

        # ---------- eligibility: treated with >=1 eligible control ----------
        # (naive O(N*M) pass; okay for moderate sizes)
        eligible_t_mask = pd.Series(False, index=T.index, dtype=bool)
        n_cands_series  = pd.Series(0,     index=T.index, dtype=int)
        
        for i, r in T.iterrows():
            ok = (
                (np.abs(C["logit"] - r["logit"]) <= caliper) &
                (np.abs(C["TimeSinceBaselineDays"] - r["TimeSinceBaselineDays"]) <= time_tolerance_days)
            )
            n = int(ok.sum())
            n_cands_series.at[i] = n
            if n > 0:
                eligible_t_mask.at[i] = True
        
        T_eligible = T.loc[eligible_t_mask].copy()
        T_eligible["n_candidates"] = n_cands_series.loc[T_eligible.index].values

        # ---------- eligibility: controls eligible for >=1 treated ----------
        # Build by symmetry (this is faster using broadcasting bins, but keep it simple/clear)
        eligible_c_mask = np.zeros(len(C), dtype=bool)
        counts_c = np.zeros(len(C), dtype=int)

        for _, r in T.iterrows():
            ok = (
                (np.abs(C["logit"] - r["logit"]) <= caliper) &
                (np.abs(C["TimeSinceBaselineDays"] - r["TimeSinceBaselineDays"]) <= time_tolerance_days)
            ).values
            eligible_c_mask |= ok
            counts_c += ok.astype(int)

        C_eligible = C.loc[eligible_c_mask].copy()
        C_eligible["n_treated_that_match"] = counts_c[eligible_c_mask]

        # ---------- save ----------
        def _out_path(in_path: str) -> str:
            base = os.path.basename(in_path)
            folder = os.path.dirname(in_path) if out_dir is None else out_dir
            return os.path.join(folder, f"eligible_{base}")

        out_custom = _out_path(custom_csv)
        out_control = _out_path(control_csv)
        T_eligible.to_csv(out_custom, index=False)
        C_eligible.to_csv(out_control, index=False)

        # ---------- report ----------
        msg = (
            f"[Eligibility summary]\n"
            f"  Treated with ≥1 eligible control: {len(T_eligible)} / {len(T)} "
            f"({len(T_eligible)/max(1,len(T)):.1%})\n"
            f"  Controls eligible for ≥1 treated: {len(C_eligible)} / {len(C)} "
            f"({len(C_eligible)/max(1,len(C)):.1%})\n"
            f"  Time caliper: ±{time_tolerance_days} days\n"
            f"  Logit caliper: {'ON' if use_logit_caliper else 'OFF'} "
            f"(SD(logit)={sd_logit:.3f}, multiplier={caliper_multiplier}, width={caliper if np.isfinite(caliper) else float('inf'):.3f})\n"
            f"  Saved:\n"
            f"    - {out_custom}\n"
            f"    - {out_control}"
        )
        print(msg)

        return {
            "treated_total": int(len(T)),
            "treated_eligible": int(len(T_eligible)),
            "control_total": int(len(C)),
            "control_eligible": int(len(C_eligible)),
            "sd_logit": sd_logit,
            "caliper_width": float(caliper),
            "time_tolerance_days": int(time_tolerance_days),
            "eligible_custom_path": out_custom,
            "eligible_control_path": out_control,
        }

    # ---------- Optional: example main ----------
if __name__ == "__main__":
    #path_ppmi = '/home/georg-tirpitz/Documents/Neuromodulation/Parkinson_PSM/PPMI'
    #csv_path_ledd  = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/data/MoCA/level2/moca_ledd.csv"
    #csv_path_updrs     = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/data/MoCA/level2/moca_updrs_joined.csv"
    #std_map   = '/home/georg-tirpitz/Documents/PD-PropensityMatching/covariate_names.csv'
    path_ppmi =  "/home/georg/Documents/Neuromodulation/PPMI"
    csv_path_ledd  = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/data/MoCA/level2/moca_ledd.csv"
    csv_path_updrs     = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/data/MoCA/level2/moca_updrs_joined.csv"
    std_map   = "covariate_names.csv"
    ppmi_data = Data(path_ppmi, foldertype="PPMI", covariate_names=std_map)

    # DBS
    ppmi_noU = ppmi_data.match_dbs(csv_path_updrs, quest="moca", STN=True, use_updrs=False)
    ppmi_U   = ppmi_data.match_dbs(csv_path_updrs, quest="moca", STN=True, use_updrs=True)

    #
    ppmi_noU = ppmi_noU[ppmi_noU["TimeSinceSurgery"] >= 0.6]
    ppmi_U   = ppmi_U[ppmi_U["TimeSinceSurgery"] >= 0.6]
    ppmi_noU.to_csv("ppmi_ledd_nu.csv", index=False)
    ppmi_U.to_csv("ppmi_ledd.csv", index=False)
    # Non-DBS
    #medication_group = ppmi_data.match_non_dbs(
    #    csv_path_updrs,
    #    quest="moca",
    #    id_column="PATNO",
    #    time_tolerance_days=120,
    #    logit_caliper_sd=-1,
    #    use_updrs=True,
    #    replace=False,
    #    random_state=42
    #)
    #medication_group["custom"].to_csv("matched_custom.csv", index=False)
    #medication_group["ppmi"].to_csv("matched_ppmi.csv", index=False)
    #medication_group["pairs"].to_csv("matched_pairs.csv", index=False)
    ## Example: lost PATNOs
    #import matplotlib.pyplot as plt
#
    #plt.figure(figsize=(8, 4))
    #plt.hist(medication_group["custom"]["propensity"], bins=20, alpha=0.7, label="Custom")
    #plt.hist(medication_group["ppmi"]["propensity"], bins=20, alpha=0.7, label="PPMI")
    #plt.axvline(0.5, color="red", linestyle="--", label="Ideal Match Region")
    #plt.title("Propensity Score Distributions")
    #plt.xlabel("Propensity Score")
    #plt.ylabel("Count")
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig("propensity_distributions.png", dpi=300)
    #plt.hist(medication_group["custom"]["logit"], bins=20, alpha=0.5, label="Custom")
    #plt.hist(medication_group["ppmi"]["logit"], bins=20, alpha=0.5, label="PPMI")
    #plt.legend()
    #plt.title("Logit Propensity Scores")
    #plt.savefig("logit_propensity_distributions.png", dpi=300)
#
    #moca_subscores = [f"{col}_sum_pre" for col in ppmi_data.MOCA_CATEGORY_MAP]
#
    #for col in moca_subscores + ["UPDRS_on", "TimeSinceDiag", "TimeSinceSurgery", "AGE_AT_BASELINE",]:
    #    plt.figure()
    #    plt.hist(medication_group["custom"][col], alpha=0.6, label="Custom", bins=15)
    #    plt.hist(medication_group["ppmi"][col], alpha=0.6, label="PPMI", bins=15)
    #    plt.title(f"{col} Distribution")
    #    plt.legend()
    #    plt.savefig(f"{col}_distribution.png", dpi=300)
    #    plt.show()
#
    #lost_patnos = sorted(set(ppmi_noU["PATNO"].unique()) - set(ppmi_U["PATNO"].unique()))
    #out_csv = "lost_patnos_when_using_updrs.csv"
    #pd.Series(lost_patnos, name="PATNO").to_csv(out_csv, index=False)
#
    #custom_csv = "matched_custom.csv"   # treated cohort file
    #ppmi_csv   = "matched_ppmi.csv"     # control cohort file
#
    #summary = ppmi_data.save_eligible_under_calipers(
    #    custom_csv,
    #    ppmi_csv,
    #    time_tolerance_days=120,     # your time window
    #    use_logit_caliper=True,      # classic choice
    #    caliper_multiplier=0.15,      
    #    custom_id_col="PATNO",
    #    control_id_col="PATNO",
    #    propensity_col="propensity", # column with p; set None if you already have a logit col
    #    logit_col=None,              # or e.g., "logit" if your csv already has logits
    #    time_col="TimeSinceBaselineDays",  # leave None to compute from a date column
    #    # out_dir="some/other/folder" # optional; defaults to each file’s own folder
    #)
    #medication_group = ppmi_data.select_medication_group(
    #csv_path=csv_path_updrs,
    #quest="moca",
    #id_column="PATNO",
    #use_updrs=True,
    #updrs_state="on",
    #top_k=8,
    #bandwidth_scale=1.0,
    #random_state=42,
    #unique_patient=True,
    #require_ledd=False,   # <— turn off LEDD requirement
    #)

    #medication_group["custom"].to_csv("med_cohort_custom.csv", index=False)
    #medication_group["ppmi"].to_csv("med_cohort_ppmi.csv", index=False)
    #medication_group["pairs"].to_csv("med_cohort_pairs.csv", index=False)
    #medication_group["balance"].to_csv("med_cohort_balance.csv")

    dist_cohort = ppmi_data.select_medication_distribution_cohort(
    csv_path=csv_path_updrs,
    quest="moca",
    id_column="PATNO",
    soft_quota_tolerance=0.5,
    use_updrs=True,
    updrs_state="both",
    unique_patient=True,          # one pair per PPMI subject
    require_ledd=False,           # skip LEDD requirement if it’s too sparse
    n_quantile_bins_per_dim=4,    # tune granularity (e.g., 4–8)
    random_state=42,
    )

    dist_cohort["ppmi"].to_csv("ppmi_medication_distribution_cohort.csv", index=False)
    ledd_dist_cohort = dist_cohort["ppmi"][dist_cohort["ppmi"]["LEDD_pre"].notna()]
    ledd_dist_cohort.to_csv("ppmi_medication_distribution_cohort_ledd.csv", index=False)
    updrs_dist_cohort = ledd_dist_cohort[ledd_dist_cohort["UPDRS_off"].notna()]
    updrs_dist_cohort.to_csv("ppmi_medication_distribution_cohort_ledd_updrs.csv", index=False)
    dist_cohort["bin_plan"].to_csv("ppmi_distribution_bin_plan.csv", index=False)
    print(dist_cohort["balance"])
    ppmi_dc = dist_cohort["ppmi"].copy()
    ppmi_dc_ledd = ledd_dist_cohort.copy()
    ppmi_dc_ledd_updrs = updrs_dist_cohort.copy()
    # Pick the age column to plot (PPMI side uses AGE_AT_BASELINE in your function)
    age_col = "AGE_AT_BASELINE" if "AGE_AT_BASELINE" in ppmi_dc.columns else (
        "AGE_MATCH" if "AGE_MATCH" in ppmi_dc.columns else None
    )

    features = [
        ("TimeSinceSurgery", "Time Since Surgery (years)"),
        ("TimeSinceDiag",    "Time Since Diagnosis (years)"),
    ]
    if age_col:
        features.append((age_col, "Age at Baseline (years)"))

    def _clean_numeric(series):
        s = pd.to_numeric(series, errors="coerce")
        return s[np.isfinite(s)]


    # Overlay: full cohort vs LEDD subset (if LEDD subset not empty)
    if len(ppmi_dc_ledd) > 0:
        for col, label in features:
            if col not in ppmi_dc_ledd.columns:
                continue
            data_full = _clean_numeric(ppmi_dc[col])
            data_ledd = _clean_numeric(ppmi_dc_ledd[col])
            if len(data_full) == 0 or len(data_ledd) == 0:
                continue

            plt.figure(figsize=(7.5, 4.8))
            # Set common bin edges for a fair overlay
            lo = np.nanmin([data_full.min(), data_ledd.min()])
            hi = np.nanmax([data_full.max(), data_ledd.max()])
            bins = np.linspace(lo, hi, 30)

            plt.hist(data_full, bins=bins, alpha=0.55, label="All PPMI", density=False)
            plt.hist(data_ledd, bins=bins, alpha=0.55, label="LEDD subset", density=False)
            plt.xlabel(label)
            plt.ylabel("Count")
            plt.title(f"Overlay: {label} — All vs LEDD subset")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(f"ppmi_distcohort_hist_overlay_{col}.png", dpi=200)
            plt.close()
