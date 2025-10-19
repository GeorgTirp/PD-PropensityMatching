import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from icecream import ic
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Callable, Dict, List, Optional, Set, Tuple
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
                    mask = ps_norm.str_contains = ps_norm.str.contains(r"\bON\b", na=False)

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
        window_years: float = 1.0,
        min_separation_days: int = 90,
        require_updrs: bool = False,
    ) -> pd.DataFrame:
        """
        For each PPMI PATNO, generate *all* valid (baseline, follow-up) pairs whose
        spacing is centred around `target_years` within ± `window_years`.

        - baseline := first available visit per patient
        - follow-up := any later visit satisfying:
              days_between >= min_separation_days
              and |days_between - target_years*365.25| <= window_years*365.25
        - If `require_updrs` is True, keep only patients with baseline UPDRS_on.

        Returns a wide df with:
          PATNO, TEST_DATUM_pre, TEST_DATUM_post,
          TimeSinceBaselineDays (at baseline),
          TimeSinceDiag (at baseline),
          TimeSinceSurgery (years between base and follow),
          abs_diff_to_target_days,
          UPDRS_on (from baseline),
          MoCA_*_sum_pre/post  (if available in long df)
        """
        if df is None or df.empty:
            return pd.DataFrame()

        pairs = []
        tgt_days = int(round(target_years * 365.25))
        win_days = int(round(window_years * 365.25))

        for patno, visits in df.groupby("PATNO"):
            v = visits.sort_values("TEST_DATUM").copy()
            if v.empty:
                continue

            base = v.iloc[0]
            if require_updrs and pd.isna(base.get("UPDRS_on", np.nan)):
                continue

            # Candidate follow-ups after baseline
            later = v.iloc[1:].copy()
            if later.empty:
                continue

            later["gap_days"] = (later["TEST_DATUM"] - base["TEST_DATUM"]).dt.days
            later = later[(later["gap_days"] >= min_separation_days)]
            if later.empty:
                continue

            later["abs_diff_to_target_days"] = (later["gap_days"] - tgt_days).abs()
            later = later[later["abs_diff_to_target_days"] <= win_days]
            if later.empty:
                continue

            # Emit *all* eligible pairs; downstream matching can pick best
            for _, follow in later.iterrows():
                time_years = follow["gap_days"] / 365.25
                row = {
                    "PATNO": patno,
                    "TEST_DATUM_pre":  base["TEST_DATUM"],
                    "TEST_DATUM_post": follow["TEST_DATUM"],
                    "TimeSinceBaselineDays": base.get("TimeSinceBaselineDays", np.nan),
                    "TimeSinceDiag":        base.get("TimeSinceDiag", np.nan),
                    "TimeSinceSurgery":     time_years,
                    "abs_diff_to_target_days": float(follow["abs_diff_to_target_days"]),
                    "UPDRS_on": base.get("UPDRS_on", np.nan),
                }

                # Carry MoCA domain sums (pre/post)
                for c in base.index:
                    if c.startswith("MoCA_") and c.endswith("_sum"):
                        row[f"{c}_pre"] = base[c]
                        if c in follow:
                            row[f"{c}_post"] = follow[c]

                pairs.append(row)

        out = pd.DataFrame(pairs)
        # Useful sort (closest to target first)
        if not out.empty and "abs_diff_to_target_days" in out.columns:
            out = out.sort_values(["PATNO", "abs_diff_to_target_days", "TEST_DATUM_post"]).reset_index(drop=True)
        return out

    def iterative_propensity_matching(
        custom_model: pd.DataFrame,
        ppmi_model: pd.DataFrame,
        time_tolerance_days: float,
        caliper_logit: float,
        replace: bool = False,
        batch_size: int = 500,
        random_state: int = 42,
        ) ->     tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Iterative *selection* using precomputed 'logit' (no re-training).
        Respects hard time caliper and a hard logit caliper.
        Preserves original PPMI row identity via '__orig_ix'.
        Ranking = |logit diff|, then |time diff|, then PATNO/TEST_DATUM.
        """

        rng = np.random.default_rng(random_state)

        # Ensure we can map picks back to the original PPMI rows
        if "__orig_ix" not in ppmi_model.columns:
            ppmi_model = ppmi_model.copy()
            ppmi_model["__orig_ix"] = ppmi_model.index

        available_ix = list(ppmi_model.index)   # working index in ppmi_model
        treated_order = custom_model.sort_values(["TimeSinceBaselineDays", custom_model.columns[0]]).index.tolist()

        matched_treated_ix: list[int] = []
        matched_control_orig_ix: list[int] = []
        pair_rows: list[dict] = []

        for start in range(0, len(treated_order), batch_size):
            block = treated_order[start:start + batch_size]

            for ci in block:
                if not available_ix and not replace:
                    break

                trow = custom_model.loc[ci]
                pool = ppmi_model.loc[available_ix] if not replace else ppmi_model

                # 1) hard time caliper
                cand = pool[(pool["TimeSinceBaselineDays"] - trow["TimeSinceBaselineDays"]).abs() <= float(time_tolerance_days)].copy()
                if cand.empty:
                    continue

                # 2) hard logit caliper
                cand = cand[(cand["logit"] - trow["logit"]).abs() <= float(caliper_logit)].copy()
                if cand.empty:
                    continue

                # 3) rank by closeness
                cand["logit_diff"] = (cand["logit"] - trow["logit"]).abs()
                cand["time_diff"]  = (cand["TimeSinceBaselineDays"] - trow["TimeSinceBaselineDays"]).abs()

                # Optional extra tie-breakers if present
                if "TimeSinceDiag" in cand.columns:
                    diag_diff = np.abs(cand["TimeSinceDiag"] - trow.get("TimeSinceDiag", np.nan))
                    cand["diag_diff"] = np.nan_to_num(diag_diff, nan=np.inf)
                else:
                    cand["diag_diff"] = np.inf

                updrs_col = "UPDRS_on" if "UPDRS_on" in cand.columns else ("UPDRS_off" if "UPDRS_off" in cand.columns else None)
                if updrs_col is not None:
                    u_diff = np.abs(cand[updrs_col] - trow.get(updrs_col, np.nan))
                    cand["updrs_diff"] = np.nan_to_num(u_diff, nan=np.inf)
                else:
                    cand["updrs_diff"] = np.inf

                cand = cand.sort_values(["logit_diff", "time_diff", "diag_diff", "updrs_diff", "PATNO", "TEST_DATUM"])

                chosen_pi = int(cand.index[0])
                chosen = cand.loc[chosen_pi]

                matched_treated_ix.append(ci)
                matched_control_orig_ix.append(int(chosen["__orig_ix"]))
                pair_rows.append({
                    "custom_index": int(ci),
                    "ppmi_index": int(chosen_pi),              # index in the working ppmi_model
                    "ppmi_orig_index": int(chosen["__orig_ix"]),  # original index for recovery
                    "time_diff_days": float(chosen["time_diff"]),
                    "logit_diff": float(chosen["logit_diff"]),
                })

                if not replace and chosen_pi in available_ix:
                    available_ix.remove(chosen_pi)

        # Build outputs
        treated_matches = custom_model.loc[matched_treated_ix].copy().reset_index(drop=True)
        control_matches = ppmi_model.set_index("__orig_ix").loc[matched_control_orig_ix].copy().reset_index(drop=True)
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
            random_state: int | None = None,  # kept for API compatibility; unused (deterministic)
            logit_caliper_sd: float = np.inf,    # if == -1 -> use iterative selector path
        ) -> Dict[str, pd.DataFrame]:
        """
        Safe non-DBS matching:
        - Propensity uses ONLY pre-treatment covariates (UPDRS_{state} + MoCA_sum_pre if present).
        - Hard caliper on visit-time alignment (± time_tolerance_days).
        - Logit-propensity caliper (± logit_caliper_sd * SD(logit)); unmatched treated visits are skipped.
        - Deterministic greedy 1:1 matching (optionally with replacement), or iterative selector when flag set.
        - Returns matched rows + pair diagnostics + SMD balance table + imputation summary.

        NOTE: No MoCA outcomes are used to *fit* propensity; they’re just carried through.
        """

        # --- tiny local helper so this function is self-contained ---
        def _coalesce_visit_date(df: pd.DataFrame,
                                 out_col: str = "TEST_DATUM",
                                 prefer: tuple[str, ...] = ("TEST_DATUM", "EXAMDT", "INFODT"),
                                 dayfirst: bool = True) -> pd.DataFrame:
            df = df.copy()
            for c in prefer:
                if c in df.columns and df[c].notna().any():
                    parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)
                    if parsed.notna().any():
                        df[out_col] = parsed
                        break
            return df

        state = updrs_state.lower().strip()
        if state not in {"off", "on"}:
            raise ValueError("updrs_state must be 'off' or 'on'")
        updrs_col = f"UPDRS_{state}"

        # ---- Load custom cohort, normalize, compute timing ----
        custom_df = pd.read_csv(csv_path)

        # Only strip _pre from known non-MoCA UPDRS columns
        columns_to_rename = {}
        for c in custom_df.columns:
            if c.startswith("UPDRS") and c.endswith("_pre"):
                columns_to_rename[c] = c.replace("_pre", "")
        custom_df = custom_df.rename(columns=columns_to_rename)
        custom_df = custom_df.rename(columns={"MDS_UPDRS_III_sum_ON": "UPDRS_ON"})
        custom_df = self._aggregate_custom_moca_columns(custom_df)

        if "UPDRS_ON" in custom_df.columns and "UPDRS_on" not in custom_df.columns:
            custom_df["UPDRS_on"] = custom_df["UPDRS_ON"]

        if "TimeSinceDiagYears" in custom_df.columns and "TimeSinceDiag" not in custom_df.columns:
            custom_df["TimeSinceDiag"] = custom_df["TimeSinceDiagYears"]
        if "TimeSinceDiagYears" in custom_df.columns and "TimeSinceDiag" in custom_df.columns:
            custom_df.drop(columns=["TimeSinceDiagYears"], inplace=True)

        # Coalesce/ensure TEST_DATUM on the custom cohort (handles files that only have INFODT)
        custom_df = _coalesce_visit_date(custom_df, out_col="TEST_DATUM",
                                         prefer=("TEST_DATUM", "EXAMDT", "INFODT"))
        # Recover TEST_DATUM if only OP_DATUM + TimeSinceSurgery present
        if "TEST_DATUM" not in custom_df.columns or custom_df["TEST_DATUM"].isna().all():
            if "OP_DATUM" in custom_df.columns and "TimeSinceSurgery" in custom_df.columns:
                custom_df["OP_DATUM"] = pd.to_datetime(custom_df["OP_DATUM"], errors="coerce")
                custom_df["TEST_DATUM"] = custom_df.apply(
                    lambda r: r["OP_DATUM"] + pd.DateOffset(days=int(pd.to_numeric(r.get("TimeSinceSurgery"), errors="coerce") * 365.25))
                    if pd.notna(r.get("OP_DATUM")) and pd.notna(pd.to_numeric(r.get("TimeSinceSurgery"), errors="coerce"))
                    else pd.NaT,
                    axis=1
                )

        # Normalize id & dates
        custom_df.rename(columns={"Pat_ID": "PATNO"}, inplace=True)  # allow earlier exports
        custom_df = safe_parse_dates(custom_df, cols=[c for c in ("TEST_DATUM", "OP_DATUM") if c in custom_df.columns],
                                     dayfirst=True, report=False)
        if id_column not in custom_df.columns:
            raise ValueError("Provide id_column for the custom cohort.")
        custom_df = custom_df.dropna(subset=[id_column, "TEST_DATUM"]).copy()
        custom_df[id_column] = custom_df[id_column].astype(str)
        custom_df.sort_values([id_column, "TEST_DATUM"], inplace=True)

        first_test = custom_df.groupby(id_column)["TEST_DATUM"].transform("min")
        custom_df["TimeSinceBaselineDays"]  = (custom_df["TEST_DATUM"] - first_test).dt.days
        custom_df["TimeSinceBaselineYears"] = custom_df["TimeSinceBaselineDays"] / 365.25

        # ---- Guard: only MoCA is supported downstream ----
        if quest.lower() != "moca":
            raise NotImplementedError("Non-DBS matching currently supports quest='moca' only.")

        # ---- Load PPMI (DBS=False) and build per-visit frame ----
        raw = self.load_ppmi()
        ppmi_cd = self.convert_to_standard_keys(raw, DBS=False)

        # Prepare MoCA+demo and coalesce visit date to TEST_DATUM (INFODT/EXAMDT → TEST_DATUM)
        ppmi_df = self._prepare_moca_with_demo(ppmi_cd)
        if ppmi_df.empty:
            raise ValueError("PPMI MoCA dataset is empty after preprocessing.")
        ppmi_df = _coalesce_visit_date(ppmi_df, out_col="TEST_DATUM",
                                       prefer=("TEST_DATUM", "EXAMDT", "INFODT"))

        # Exclude PPMI participants who had surgery
        dbs_df = ppmi_cd.get("dbs")
        if dbs_df is not None and "PATNO" in dbs_df.columns:
            ppmi_df = ppmi_df[~ppmi_df["PATNO"].isin(set(dbs_df["PATNO"].unique()))]

        ppmi_df = safe_parse_dates(ppmi_df, cols=["TEST_DATUM", "DIAG_DATE"], dayfirst=True, report=False)
        ppmi_df = ppmi_df.dropna(subset=["PATNO", "TEST_DATUM"]).copy()
        ppmi_df.sort_values(["PATNO", "TEST_DATUM"], inplace=True)
        ppmi_first = ppmi_df.groupby("PATNO")["TEST_DATUM"].transform("min")
        ppmi_df["TimeSinceBaselineDays"]  = (ppmi_df["TEST_DATUM"] - ppmi_first).dt.days
        ppmi_df["TimeSinceBaselineYears"] = ppmi_df["TimeSinceBaselineDays"] / 365.25

        # Compute TimeSinceDiag (fallback to baseline if DIAG_DATE missing)
        if "DIAG_DATE" in ppmi_df.columns:
            ppmi_df["TimeSinceDiag"] = ((ppmi_df["TEST_DATUM"] - ppmi_df["DIAG_DATE"]).dt.days / 365.25)
        miss_diag = ppmi_df["TimeSinceDiag"].isna() if "TimeSinceDiag" in ppmi_df.columns else None
        if miss_diag is None or miss_diag.any():
            ppmi_df.loc[miss_diag.fillna(True), "TimeSinceDiag"] = (ppmi_df.loc[miss_diag.fillna(True), "TimeSinceBaselineDays"] / 365.25)
        if "TimeSinceDiagYears" in ppmi_df.columns and "TimeSinceDiag" in ppmi_df.columns:
            ppmi_df.drop(columns=["TimeSinceDiagYears"], inplace=True)

        # ---- Attach UPDRS_{state} per visit (EVENT_ID first; fallback by unified TEST_DATUM) ----
        if use_updrs and "mds_updrs" in ppmi_cd:
            up = ppmi_cd["mds_updrs"].get("mds_updrs3")
            if up is not None and not up.empty:
                # coalesce to TEST_DATUM on the UPDRS table as well
                up = _coalesce_visit_date(up, out_col="TEST_DATUM",
                                          prefer=("TEST_DATUM", "EXAMDT", "INFODT"))
                on_tbl = self._get_updrs3_by_state(up, state=state)  # returns PATNO, TEST_DATUM, [EVENT_ID?], UPDRS_*
                on_tbl = _coalesce_visit_date(on_tbl, out_col="TEST_DATUM",
                                              prefer=("TEST_DATUM", "EXAMDT", "INFODT"))
                on_tbl = safe_parse_dates(on_tbl, cols=["TEST_DATUM"], dayfirst=True, report=False)

                # 1) EVENT_ID exact join (preferred, only if both have it)
                if "EVENT_ID" in ppmi_df.columns and "EVENT_ID" in on_tbl.columns:
                    ppmi_df = ppmi_df.merge(
                        on_tbl[["PATNO", "EVENT_ID", "TEST_DATUM", updrs_col]],
                        on=["PATNO", "EVENT_ID"],
                        how="left",
                        suffixes=("", "_ONJ")
                    )
                else:
                    # ensure column exists for downstream filters
                    if updrs_col not in ppmi_df.columns:
                        ppmi_df[updrs_col] = np.nan

                # 2) Fallback by date proximity for rows still missing UPDRS (now TEST_DATUM is guaranteed)
                need = ppmi_df[ppmi_df[updrs_col].isna()][["PATNO", "TEST_DATUM"]].copy()
                if not need.empty:
                    tmp = on_tbl.rename(columns={"TEST_DATUM": "UPDRS_TEST_DATUM"})
                    fb = need.merge(tmp[["PATNO", "UPDRS_TEST_DATUM", updrs_col]], on="PATNO", how="left")
                    fb["d"] = (fb["UPDRS_TEST_DATUM"] - fb["TEST_DATUM"]).dt.days
                    fb = fb.loc[fb["d"].abs() <= 365].copy()  # generous ±1y
                    if not fb.empty:
                        fb["abs_d"] = fb["d"].abs()
                        fb["past_first"] = (fb["d"] > 0).astype(int)
                        fb = fb.sort_values(["PATNO", "TEST_DATUM", "abs_d", "past_first"]).drop_duplicates(
                            subset=["PATNO", "TEST_DATUM"], keep="first")
                        ppmi_df = ppmi_df.merge(
                            fb[["PATNO", "TEST_DATUM", updrs_col]],
                            on=["PATNO", "TEST_DATUM"], how="left", suffixes=("", "_FB")
                        )
                        mask = ppmi_df[updrs_col].isna() & ppmi_df[f"{updrs_col}_FB"].notna()
                        ppmi_df.loc[mask, updrs_col] = ppmi_df.loc[mask, f"{updrs_col}_FB"]
                        ppmi_df.drop(columns=[f"{updrs_col}_FB"], inplace=True, errors="ignore")

                print(f"[DEBUG] PPMI visits: {len(ppmi_df)} | UPDRS {state.upper()} attached: {ppmi_df[updrs_col].notna().sum()} "
                      f"({ppmi_df[updrs_col].notna().mean():.2%})")

        # ---- STRICT ON-only pool with per-patient coverage ≥50% ----
        if use_updrs:
            if "UPDRS_on" not in ppmi_df.columns:
                raise ValueError("UPDRS_on column is missing after attachment – cannot proceed with ON-only matching.")

            ppmi_df["has_on"] = ppmi_df["UPDRS_on"].notna()
            on_cov = ppmi_df.groupby("PATNO")["has_on"].mean()
            eligible_patnos = on_cov[on_cov >= 0.50].index

            ppmi_df = ppmi_df[ppmi_df["PATNO"].isin(eligible_patnos)].copy()
            ppmi_df = ppmi_df[ppmi_df["has_on"]].copy()
            ppmi_df.drop(columns=["has_on"], inplace=True)
            if "UPDRS_on" in custom_df.columns:
                custom_df = custom_df[custom_df["UPDRS_on"].notna()].copy()

            print(f"[DEBUG] ON-only filter: eligible PPMI patients ≥50% ON: {len(eligible_patnos)} "
                  f"| visits kept: {len(ppmi_df)} | mean ON coverage among eligibles: {on_cov.loc[eligible_patnos].mean():.2%}")

        # ---- Build propensity inputs (ONLY pre-treatment covariates) ----
        ppmi_df_pairs = self._build_ppmi_timeline_pairs(ppmi_df)   # produces *_pre / *_post, base TEST_DATUM_pre
        ppmi_model = ppmi_df_pairs.copy()
        ppmi_model = ppmi_model.rename(columns={"TEST_DATUM_pre": "TEST_DATUM"})  # use baseline date as TEST_DATUM

        covars = ["TimeSinceDiag", "TimeSinceSurgery"]
        if use_updrs and updrs_col in ppmi_model.columns:
            covars.append(updrs_col)

        # baseline MoCA (if present)
        moca_baseline_cols = [c for c in ["MoCA_sum_pre"] if c in ppmi_model.columns]
        covars.extend(moca_baseline_cols)

        # Ensure covariates exist in custom_df (placeholder if missing)
        for c in covars:
            if c not in custom_df.columns:
                custom_df[c] = np.nan

        # Drop to rows having time (we'll impute covariates inside the pipeline)
        custom_model = custom_df.dropna(subset=["TimeSinceBaselineDays"]).copy()
        if custom_model.empty or ppmi_model.empty:
            raise ValueError("No rows with usable timing in one of the cohorts.")

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
            ("logit", LogisticRegression(solver="liblinear", C=1.0, max_iter=2000,
                                         class_weight="balanced", random_state=42))
        ])

        # Design matrices BEFORE fitting
        X_custom = custom_model[covars]
        X_ppmi   = ppmi_model[covars]
        X_all    = pd.concat([X_custom, X_ppmi], axis=0, ignore_index=True)
        y_all    = np.concatenate([np.ones(len(X_custom), dtype=int),
                                   np.zeros(len(X_ppmi), dtype=int)])

        print("[match_non_dbs] Matching covariates used in propensity model:")
        for c in covars:
            n_custom = custom_df[c].notna().sum()
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
        logit_fn = lambda p: np.log(np.clip(p, eps, 1 - p, eps) / np.clip(1 - p, eps, 1 - p, eps))
        # small fix: clip both numerator and denominator separately (done above)
        l_custom = np.log(np.clip(p_custom, eps, 1 - eps) / np.clip(1 - p_custom, eps, 1 - eps))
        l_ppmi   = np.log(np.clip(p_ppmi,   eps, 1 - eps) / np.clip(1 - p_ppmi,   eps, 1 - eps))
        sd_logit = np.std(np.concatenate([l_custom, l_ppmi]))
        caliper  = (logit_caliper_sd * sd_logit) if sd_logit > 0 else np.inf

        custom_model = custom_model.assign(propensity=p_custom, logit=l_custom)
        ppmi_model   = ppmi_model.assign(propensity=p_ppmi,   logit=l_ppmi)

        # Utility helpers used for both paths
        def keep_cols(df, who: str):
            keep = [c for c in df.columns if c.startswith("MoCA_")]  # all MoCA
            for c in ["TimeSinceDiag", updrs_col, updrs_col.upper() if updrs_col.upper()!=updrs_col else updrs_col,
                      "PATNO", id_column, "TEST_DATUM", "propensity", "logit",
                      "matched_ppmi_patno", "matched_ppmi_index", "matched_custom_id", "matched_custom_index"]:
                if c in df.columns and c not in keep:
                    keep.append(c)
            keep = list(dict.fromkeys(keep))
            out = df[keep].copy()
            out["COHORT"] = who
            return out

        def _standardize_moca_suffixes(df: pd.DataFrame) -> pd.DataFrame:
            newcols = {}
            for c in df.columns:
                if c.startswith("MoCA_"):
                    if c.endswith("_pre") and not c.endswith("_sum_pre"):
                        newcols[c] = c.replace("_pre", "_sum_pre")
                    elif c.endswith("_post") and not c.endswith("_sum_post"):
                        newcols[c] = c.replace("_post", "_sum_post")
            return df.rename(columns=newcols)

        # ---- Path 1: Iterative selector (no re-training) ----
        if logit_caliper_sd == -1:
            print("[match_non_dbs] Using iterative selection with global propensities")
            if "__orig_ix" not in ppmi_model.columns:
                ppmi_model["__orig_ix"] = ppmi_model.index

            treated_matches, control_matches, pair_df = iterative_propensity_matching(
                custom_model=custom_model,
                ppmi_model=ppmi_model,
                time_tolerance_days=time_tolerance_days,
                caliper_logit=sd_logit * 0.2 if np.isfinite(sd_logit) and sd_logit > 0 else np.inf,  # default 0.2*SD
                replace=replace,
                batch_size=500,
                random_state=42,
            )

            treated_matches = self._rename_moca_pre_post(treated_matches)
            control_matches = self._rename_moca_pre_post(control_matches)
            treated_matches = _standardize_moca_suffixes(treated_matches)
            control_matches = _standardize_moca_suffixes(control_matches)

            treated_matches = keep_cols(treated_matches, "Custom")
            control_matches = keep_cols(control_matches, "PPMI")

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

            # 2) logit caliper (hard)
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

            cand = cand.sort_values(["prop_diff", "time_diff", "diag_diff", "updrs_diff", "PATNO", "TEST_DATUM"])
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
        matched_custom = _standardize_moca_suffixes(matched_custom)
        matched_ppmi   = _standardize_moca_suffixes(matched_ppmi)

        # Keep only needed columns (+ diagnostics)
        matched_custom = keep_cols(matched_custom, "Custom")
        matched_ppmi   = keep_cols(matched_ppmi, "PPMI")

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


# ---------- Optional: example main ----------
if __name__ == "__main__":
    #path_ppmi = '/home/georg-tirpitz/Documents/Neuromodulation/Parkinson_PSM/PPMI'
    #csv_path  = '/home/georg-tirpitz/Documents/Neuromodulation/ddbm/out/MOCA/level2/moca_ledd.csv'
    #std_map   = '/home/georg-tirpitz/Documents/PD-PropensityMatching/covariate_names.csv'
    path_ppmi =  "/home/georg/Documents/Neuromodulation/PPMI"
    csv_path  = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/data/MoCA/level2/moca_ledd.csv"
    std_map   = "covariate_names.csv"
    ppmi_data = Data(path_ppmi, foldertype="PPMI", covariate_names=std_map)

    # DBS
    ppmi_noU = ppmi_data.match_dbs(csv_path, quest="moca", STN=True, use_updrs=False)
    ppmi_U   = ppmi_data.match_dbs(csv_path, quest="moca", STN=True, use_updrs=True)

    #
    ppmi_noU = ppmi_noU[ppmi_noU["TimeSinceSurgery"] >= 0.6]
    ppmi_U   = ppmi_U[ppmi_U["TimeSinceSurgery"] >= 0.6]
    ppmi_noU.to_csv("ppmi_ledd_nu.csv", index=False)
    ppmi_U.to_csv("ppmi_ledd.csv", index=False)
    # Non-DBS
    medication_group = ppmi_data.match_non_dbs(
        csv_path,
        quest="moca",
        id_column="PATNO",
        time_tolerance_days=120,
        use_updrs=True,
        replace=False,
        random_state=42
    )
    medication_group["custom"].to_csv("matched_custom.csv", index=False)
    medication_group["ppmi"].to_csv("matched_ppmi.csv", index=False)
    medication_group["pairs"].to_csv("matched_pairs.csv", index=False)
    # Example: lost PATNOs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.hist(medication_group["custom"]["propensity"], bins=20, alpha=0.7, label="Custom")
    plt.hist(medication_group["ppmi"]["propensity"], bins=20, alpha=0.7, label="PPMI")
    plt.axvline(0.5, color="red", linestyle="--", label="Ideal Match Region")
    plt.title("Propensity Score Distributions")
    plt.xlabel("Propensity Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("propensity_distributions.png", dpi=300)
    plt.hist(medication_group["custom"]["logit"], bins=20, alpha=0.5, label="Custom")
    plt.hist(medication_group["ppmi"]["logit"], bins=20, alpha=0.5, label="PPMI")
    plt.legend()
    plt.title("Logit Propensity Scores")
    plt.savefig("logit_propensity_distributions.png", dpi=300)

    moca_subscores = [f"{col}_sum_pre" for col in ppmi_data.MOCA_CATEGORY_MAP]

    for col in moca_subscores + ["UPDRS_on", "TimeSinceDiag",]:
        plt.figure()
        plt.hist(medication_group["custom"][col], alpha=0.6, label="Custom", bins=15)
        plt.hist(medication_group["ppmi"][col], alpha=0.6, label="PPMI", bins=15)
        plt.title(f"{col} Distribution")
        plt.legend()
        plt.savefig(f"{col}_distribution.png", dpi=300)
        plt.show()

    lost_patnos = sorted(set(ppmi_noU["PATNO"].unique()) - set(ppmi_U["PATNO"].unique()))
    out_csv = "lost_patnos_when_using_updrs.csv"
    pd.Series(lost_patnos, name="PATNO").to_csv(out_csv, index=False)
    print(f"Saved {len(lost_patnos)} lost PATNOs to: {out_csv}")
