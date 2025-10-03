import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from icecream import ic
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    MOCA_DATA_COLS = [
        'MoCA_Executive',
        'MoCA_Benennen',
        'MoCA_Aufmerksamkeit_Zahlenliste',
        'MoCA_Aufmerksamkeit_Buchstabenliste',
        'MoCA_Aufmerksamkeit_Abziehen',
        'MoCA_Sprache_Wiederholen',
        'MoCA_Sprache_Buchstaben',
        'MoCA_Abstraktion',
        'MoCA_Erinnerung',
        'MoCA_Orientierung'
    ]

    MOCA_CATEGORIES = {
        "Aufmerksamkeit": [
            "MoCA_Aufmerksamkeit_Zahlenliste",
            "MoCA_Aufmerksamkeit_Buchstabenliste",
            "MoCA_Aufmerksamkeit_Abziehen"
        ],
        "Sprache": [
            "MoCA_Sprache_Wiederholen",
            "MoCA_Sprache_Buchstaben"
        ],
        "Benennen": ["MoCA_Benennen"],
        "Executive": ["MoCA_Executive"],
        "Abstraktion": ["MoCA_Abstraktion"],
        "Erinnerung": ["MoCA_Erinnerung"],
        "Orientierung": ["MoCA_Orientierung"]
    }

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
            diag_df["DIAG_DATE"] = diag_df["PDDXDT"]
            demo_df = pd.merge(pd.DataFrame(demo_dict), diag_df, on="PATNO", how="left")
        else:
            cd['demo'] = safe_parse_dates(cd['demo'], cols=["PDSURGDT"], dayfirst=True, report=True)
            #demo_dict['OP_DATUM'] = cd['demo']['PDSURGDT']
            demo_dict['SEX'] = cd['demo']['SEX']
            demo_dict['PATNO'] = cd['demo']['PATNO']
            diag_df = cd["diaghist"][["PATNO", "PDDXDT"]].copy()
            diag_df = safe_parse_dates(diag_df, cols=["PDDXDT"], dayfirst=True, report=True)
            diag_df["DIAG_DATE"] = diag_df["PDDXDT"]
            demo_df = pd.merge(pd.DataFrame(demo_dict), diag_df, on="PATNO", how="left")

        # ---- MoCA reshape ----
        moca_dict = {}
        moca = cd['moca']
        moca_dict['executive']           = moca.iloc[:, 5:9].sum(axis=1)
        moca_dict['naming']              = moca.iloc[:, 10:12].sum(axis=1)
        moca_dict['attention_numbers']   = moca.iloc[:, 13:14].sum(axis=1)
        moca_dict['attention_letters']   = moca.iloc[:, 15]
        moca_dict['attention_substract'] = moca.iloc[:, 16]
        moca_dict['language_rep']        = moca.iloc[:, 17]
        moca_dict['language_letters']    = moca.iloc[:, 18:19].sum(axis=1)
        moca_dict['abstraction']         = moca.iloc[:, 20]
        moca_dict['reminding']           = moca.iloc[:, 21:25].sum(axis=1)
        moca_dict['orientation']         = moca.iloc[:, 26:31].sum(axis=1)
        moca_dict['total']               = moca.iloc[:, 32]

        moca_dict = dict(zip(covariate_dict['moca'][5:16], moca_dict.values()))
        cd['moca'] = pd.concat([moca.iloc[:, :5], pd.DataFrame(moca_dict), moca.iloc[:, -2:]], axis=1)
        cd['moca'].rename(columns={'INFODT': covariate_dict['moca'][4]}, inplace=True)

        # Replace demo with prepared demo_df
        cd['demo'] = demo_df

        return cd

    # ---------- Helpers operating on provided dicts ----------
    def _augment_moca_scores(self, moca_df: pd.DataFrame) -> pd.DataFrame:
        df = moca_df.copy()
        if "MoCA_ONLY_GES" in df.columns:
            df["MoCA_sum"] = df.apply(
                lambda row: row[self.MOCA_DATA_COLS].sum(skipna=False) if pd.isna(row["MoCA_ONLY_GES"]) else row["MoCA_ONLY_GES"],
                axis=1
            )
        for cat_name, item_cols in self.MOCA_CATEGORIES.items():
            if set(item_cols).issubset(df.columns):
                df[f"MoCA_{cat_name}_sum"] = df[item_cols].sum(axis=1, skipna=False)
        return df

    def _prepare_moca_with_demo(self, complete_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        moca_df = complete_data.get("moca")
        if moca_df is None:
            return pd.DataFrame()
        moca_df = self._augment_moca_scores(moca_df)
        demo_df = complete_data.get("demo")
        if demo_df is None:
            return moca_df

        demo_keep = ["PATNO","OP_DATUM","LOCATION","DIAG_DATE","SEX","AGE_AT_OP"]
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

    # ---------- Matching: DBS ----------
    def match_dbs(
        self,
        csv_path: str,
        quest: str,
        STN: bool = True,
        use_updrs: bool = True
    ) -> pd.DataFrame:
        """
        Build DBS cohort pairs locally; no persistent state.
        """
        # Load external outcome file
        df = pd.read_csv(csv_path)
        df = df.rename(columns={c: c.replace('_pre', '') for c in df.columns if c.endswith('_pre')})
        df = df[[c for c in df.columns if not c.endswith('_post')]]

        # Load PPMI raw + convert (DBS=True)
        raw = self.load_ppmi()  # covariate_names already stored when first called
        ppmi_cd = self.convert_to_standard_keys(raw, DBS=True)

        # Merge demographic + questionnaire
        if quest.lower() != "moca":
            raise NotImplementedError("Only quest='moca' currently supported.")
        moca_with_demo = self._prepare_moca_with_demo(ppmi_cd)
        if moca_with_demo.empty:
            return moca_with_demo

        essential_cols = [
            "PATNO","OP_DATUM","TEST_DATUM","MoCA_sum","LOCATION","DIAG_DATE","SEX","AGE_AT_OP"
        ]
        available_cols = [c for c in essential_cols if c in moca_with_demo.columns]
        pc_df = moca_with_demo[available_cols].copy()

        # add category sums if present
        for cat_name in self.MOCA_CATEGORIES:
            col_name = f"MoCA_{cat_name}_sum"
            if col_name in moca_with_demo.columns:
                pc_df[col_name] = moca_with_demo[col_name]

        subset_cols = [c for c in ["PATNO", "OP_DATUM", "TEST_DATUM", "LOCATION"] if c in pc_df.columns]
        pc_df_before_merge = pc_df[subset_cols].copy()

        # Time metrics
        pc_df = safe_parse_dates(pc_df, cols=["TEST_DATUM", "OP_DATUM", "DIAG_DATE"], dayfirst=True, report=True)
        pc_df_before_merge = safe_parse_dates(pc_df_before_merge, cols=["TEST_DATUM","OP_DATUM"], dayfirst=True, report=True)

        pc_df["TimeSinceSurgery"] = (pc_df["TEST_DATUM"] - pc_df["OP_DATUM"]).dt.days / 365.25
        pc_df_before_merge["TimeSinceSurgery"] = (pc_df_before_merge["TEST_DATUM"] - pc_df_before_merge["OP_DATUM"]).dt.days / 365.25
        pc_df_fu = pc_df[pc_df["TimeSinceSurgery"] > 0].copy()
        pc_df["TimeSinceDiag"] = (pc_df["TEST_DATUM"] - pc_df["DIAG_DATE"]).dt.days / 365.25

        if STN:
            pc_df = pc_df[pc_df["LOCATION"] == "STN"]
            pc_df = pc_df.drop(columns=["LOCATION"])
            pc_df_fu = pc_df_fu[pc_df_fu["LOCATION"] == "STN"]

        if use_updrs:
            updrs_df = ppmi_cd["mds_updrs"]["mds_updrs3"]
            treated = updrs_df[updrs_df["PDMEDYN"] == 1].copy()
            treated["TEST_DATUM"] = treated["INFODT"]

            off_df = treated[(treated["OFFEXAM"] == 1) | (treated["ONEXAM"] == 0)][
                ["PATNO", "TEST_DATUM", "MDS-UPDRS_3_ONLY_Gesamt"]
            ].rename(columns={"MDS-UPDRS_3_ONLY_Gesamt": "UPDRS_OFF"})
            on_df = treated[(treated["ONEXAM"] == 1) | (treated["OFFEXAM"] == 0)][
                ["PATNO", "TEST_DATUM", "MDS-UPDRS_3_ONLY_Gesamt"]
            ].rename(columns={"MDS-UPDRS_3_ONLY_Gesamt": "UPDRS_ON"})

            merged = pd.merge(off_df, on_df, on=["PATNO", "TEST_DATUM"])
            merged["TEST_DATUM"] = pd.to_datetime(merged["TEST_DATUM"])
            merged["UPDRS_reduc"] = (merged["UPDRS_OFF"] - merged["UPDRS_ON"]) / merged["UPDRS_OFF"].dropna()
            merged = merged.rename(columns={"TEST_DATUM": "UPDRS_TEST_DATUM"})
            merged = merged[["UPDRS_reduc", "UPDRS_TEST_DATUM", "PATNO"]]

            merged = pd.merge(pc_df, merged, on="PATNO", how="inner")
            merged["UPDRSET_dist"] = (merged["UPDRS_TEST_DATUM"] - merged["TEST_DATUM"]).dt.days.abs()
            merged = merged[(merged["UPDRSET_dist"] <= 730) & (merged["UPDRS_TEST_DATUM"] < merged["OP_DATUM"])]

            pc_df = (
                merged.sort_values("UPDRSET_dist")
                .groupby(["PATNO", "TEST_DATUM"], as_index=False)
                .first()
                .dropna()
            )

        # baseline & follow-up
        pc_df_bl = pc_df[pc_df["TimeSinceSurgery"] <= 0].copy()
        pc_df_bl = pc_df_bl.drop(columns=["TimeSinceSurgery","TEST_DATUM","DIAG_DATE","UPDRS_TEST_DATUM","UPDRSET_dist"], errors="ignore")
        pc_df_fu = pc_df_fu.drop(columns=["TimeSinceDiag","UPDRS_reduc","AGE_AT_OP","OP_DATUM","SEX","DIAG_DATE","UPDRS_TEST_DATUM","UPDRSET_dist"], errors="ignore")

        suffix_cols_bl = [c for c in pc_df_bl.columns if c.startswith("MoCA_")]
        suffix_cols_fu = [c for c in pc_df_fu.columns if c.startswith("MoCA_")]

        pc_df_bl = pc_df_bl.rename(columns={c: f"{c}_pre" for c in suffix_cols_bl})
        pc_df_fu = pc_df_fu.rename(columns={c: f"{c}_post" for c in suffix_cols_fu})

        combined = pd.merge(pc_df_bl, pc_df_fu, on=["PATNO"], how="inner")
        combined["TimeSinceSurgery_abs_diff"] = (combined["TimeSinceSurgery"] - 2).abs()
        closest_followups = combined.loc[combined.groupby("PATNO")["TimeSinceSurgery_abs_diff"].idxmin()].reset_index(drop=True)
        final_df = closest_followups.drop(columns=["TimeSinceSurgery_abs_diff"])
        print(len(final_df))
        return final_df

    # ---------- Matching: non-DBS ----------
    def match_non_dbs(
        self,
        csv_path: str,
        quest: str,
        id_column: Optional[str] = None,
        time_tolerance_days: int = 120,
        use_updrs: bool = True,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:

        # Load custom cohort
        custom_df = pd.read_csv(csv_path)
        custom_df = custom_df.rename(columns={c: c.replace('_pre', '') for c in custom_df.columns if c.endswith('_pre')})

        if "TEST_DATUM" not in custom_df.columns and "OP_DATUM" in custom_df.columns and "TimeSinceSurgery" in custom_df.columns:
            custom_df["OP_DATUM"] = pd.to_datetime(custom_df["OP_DATUM"])
            custom_df["TEST_DATUM"] = custom_df.apply(
                lambda row: row["OP_DATUM"] + pd.DateOffset(days=int(row["TimeSinceSurgery"] * 365.25)),
                axis=1
            )
            custom_df["TEST_DATUM"] = custom_df["TEST_DATUM"].dt.strftime("%Y-%m-%d")

        custom_df.rename(columns={"Pat_ID": "PATNO"}, inplace=True)

        if id_column is None:
            raise ValueError("Provide id_column for the custom cohort.")

        custom_df = safe_parse_dates(custom_df, cols=["TEST_DATUM"], dayfirst=True, report=False)
        custom_df = custom_df.dropna(subset=[id_column, "TEST_DATUM"]).copy()
        custom_df[id_column] = custom_df[id_column].astype(str)
        custom_df.sort_values([id_column, "TEST_DATUM"], inplace=True)

        first_test = custom_df.groupby(id_column)["TEST_DATUM"].transform("min")
        custom_df["TimeSinceBaselineDays"] = (custom_df["TEST_DATUM"] - first_test).dt.days
        custom_df["TimeSinceBaselineYears"] = custom_df["TimeSinceBaselineDays"] / 365.25
        custom_df["VisitNumber"] = custom_df.groupby(id_column).cumcount()

        if quest.lower() != "moca":
            raise NotImplementedError("Non-DBS matching currently supports quest='moca' only.")

        # Load PPMI raw + convert for non-DBS (DBS=False)
        raw = self.load_ppmi()  # uses stored covariate_names
        ppmi_cd = self.convert_to_standard_keys(raw, DBS=False)

        ppmi_df = self._prepare_moca_with_demo(ppmi_cd)
        if ppmi_df.empty:
            raise ValueError("PPMI MoCA dataset is empty after preprocessing.")

        # Exclude PPMI subjects with surgery
        dbs_df = ppmi_cd.get("dbs")
        if dbs_df is not None and "PATNO" in dbs_df.columns:
            dbs_patnos: Set = set(dbs_df["PATNO"].unique())
            ppmi_df = ppmi_df[~ppmi_df["PATNO"].isin(dbs_patnos)]

        if ppmi_df.empty:
            raise ValueError("No eligible PPMI participants remain after excluding DBS cases.")

        ppmi_df = safe_parse_dates(ppmi_df, cols=["TEST_DATUM", "DIAG_DATE"], dayfirst=True, report=False)
        ppmi_df = ppmi_df.dropna(subset=["PATNO", "TEST_DATUM"]).copy()
        ppmi_df.sort_values(["PATNO", "TEST_DATUM"], inplace=True)

        ppmi_first = ppmi_df.groupby("PATNO")["TEST_DATUM"].transform("min")
        ppmi_df["TimeSinceBaselineDays"] = (ppmi_df["TEST_DATUM"] - ppmi_first).dt.days
        ppmi_df["TimeSinceBaselineYears"] = ppmi_df["TimeSinceBaselineDays"] / 365.25
        ppmi_df["VisitNumber"] = ppmi_df.groupby("PATNO").cumcount()
        if "DIAG_DATE" in ppmi_df.columns:
            ppmi_df["TimeSinceDiagYears"] = ((ppmi_df["TEST_DATUM"] - ppmi_df["DIAG_DATE"]).dt.days / 365.25)

        if use_updrs and "mds_updrs" in ppmi_cd:
            updrs_df = ppmi_cd["mds_updrs"].get("mds_updrs3")
            if updrs_df is not None and not updrs_df.empty:
                treated = updrs_df[updrs_df["PDMEDYN"] == 1].copy()
                treated["TEST_DATUM"] = pd.to_datetime(treated["INFODT"], errors="coerce")
                off_df = treated[(treated["OFFEXAM"] == 1) | (treated["ONEXAM"] == 0)][[
                    "PATNO", "TEST_DATUM", "MDS-UPDRS_3_ONLY_Gesamt"
                ]].rename(columns={"MDS-UPDRS_3_ONLY_Gesamt": "UPDRS_OFF"})
                on_df = treated[(treated["ONEXAM"] == 1) | (treated["OFFEXAM"] == 0)][[
                    "PATNO", "TEST_DATUM", "MDS-UPDRS_3_ONLY_Gesamt"
                ]].rename(columns={"MDS-UPDRS_3_ONLY_Gesamt": "UPDRS_ON"})

                updrs_merge = pd.merge(off_df, on_df, on=["PATNO", "TEST_DATUM"])
                if not updrs_merge.empty:
                    updrs_merge["UPDRS_reduc"] = (updrs_merge["UPDRS_OFF"] - updrs_merge["UPDRS_ON"]) / updrs_merge["UPDRS_OFF"]
                    ppmi_df = pd.merge(
                        ppmi_df,
                        updrs_merge[["PATNO", "TEST_DATUM", "UPDRS_reduc"]],
                        on=["PATNO", "TEST_DATUM"],
                        how="left"
                    )

        # Shared numeric features
        exclude_cols: Set[str] = {
            id_column, "PATNO", "TEST_DATUM", "OP_DATUM", "DIAG_DATE", "LOCATION", "SEX",
            "AGE_AT_OP", "TimeSinceBaselineDays", "TimeSinceBaselineYears", "VisitNumber", "TimeSinceDiagYears"
        }

        custom_work = custom_df.copy()
        ppmi_work = ppmi_df.copy()
        custom_numeric = self._numeric_columns(custom_work, exclude_cols)
        ppmi_numeric = self._numeric_columns(ppmi_work, exclude_cols)
        shared_features = sorted(set(custom_numeric) & set(ppmi_numeric))

        if not shared_features:
            raise ValueError("No overlapping numeric covariates found between cohorts.")

        feature_cols = list(dict.fromkeys(shared_features + ["TimeSinceBaselineYears"]))

        custom_model_df = custom_work.dropna(subset=feature_cols + ["TimeSinceBaselineDays"])
        ppmi_model_df = ppmi_work.dropna(subset=feature_cols + ["TimeSinceBaselineDays"])

        if custom_model_df.empty:
            raise ValueError("Custom cohort has no rows with complete data for matching features.")
        if ppmi_model_df.empty:
            raise ValueError("PPMI cohort has no rows with complete data for matching features.")

        # Propensity
        X_custom = custom_model_df[feature_cols]
        X_ppmi   = ppmi_model_df[feature_cols]
        X_combined = pd.concat([X_custom, X_ppmi], axis=0)
        y_combined = np.concatenate([np.ones(len(X_custom), dtype=int), np.zeros(len(X_ppmi), dtype=int)])

        propensity_model = Pipeline([
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, solver="lbfgs"))
        ])
        propensity_model.fit(X_combined, y_combined)

        custom_model_df["propensity"] = propensity_model.predict_proba(X_custom)[:, 1]
        ppmi_model_df["propensity"]   = propensity_model.predict_proba(X_ppmi)[:, 1]

        ppmi_pool = ppmi_model_df.copy()
        available_idx = list(ppmi_pool.index)
        rng = np.random.default_rng(random_state) if random_state is not None else None

        matched_custom_idx: List[int] = []
        matched_ppmi_idx: List[int] = []
        match_records: List[Dict[str, float]] = []

        tolerance = float(time_tolerance_days)

        for idx, row in custom_model_df.sort_values("TimeSinceBaselineDays").iterrows():
            if not available_idx:
                break

            pool_df = ppmi_pool.loc[available_idx]
            pool_df = pool_df.assign(
                time_diff=(pool_df["TimeSinceBaselineDays"] - row["TimeSinceBaselineDays"]).abs(),
                propensity_diff=(pool_df["propensity"] - row["propensity"]).abs()
            )

            within = pool_df[pool_df["time_diff"] <= tolerance]
            ranked = within if not within.empty else pool_df
            ranked = ranked.sort_values(["propensity_diff", "time_diff"])
            if ranked.empty:
                continue

            candidate_indices = ranked.index.to_numpy()
            if rng is not None and candidate_indices.size > 1:
                rng.shuffle(candidate_indices)

            chosen_idx = int(candidate_indices[0])
            chosen_row = ranked.loc[chosen_idx]

            matched_custom_idx.append(idx)
            matched_ppmi_idx.append(chosen_idx)
            match_records.append({
                "custom_index": int(idx),
                "custom_id": row[id_column],
                "ppmi_index": int(chosen_idx),
                "ppmi_patno": ppmi_pool.loc[chosen_idx, "PATNO"],
                "propensity_diff": float(chosen_row["propensity_diff"]),
                "time_diff_days": float(chosen_row["time_diff"]),
                "custom_time_days": float(row["TimeSinceBaselineDays"]),
                "ppmi_time_days": float(chosen_row["TimeSinceBaselineDays"])
            })

            if not replace:
                available_idx.remove(chosen_idx)

        if not matched_custom_idx:
            raise ValueError("No matches could be constructed under the current settings.")

        matched_custom = custom_model_df.loc[matched_custom_idx].copy()
        matched_ppmi   = ppmi_pool.loc[matched_ppmi_idx].copy()
        diagnostics    = pd.DataFrame(match_records)

        matched_custom["matched_ppmi_patno"] = matched_ppmi["PATNO"].values
        matched_custom["matched_ppmi_index"] = matched_ppmi_idx
        matched_ppmi["matched_custom_id"]    = matched_custom[id_column].values
        matched_ppmi["matched_custom_index"] = matched_custom_idx
        matched_ppmi["propensity_diff"]      = diagnostics["propensity_diff"].values
        matched_ppmi["time_diff_days"]       = diagnostics["time_diff_days"].values

        return {
            "custom": matched_custom.reset_index(drop=True),
            "ppmi": matched_ppmi.reset_index(drop=True),
            "pairs": diagnostics.reset_index(drop=True)
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
    path_ppmi = '/home/georg-tirpitz/Documents/Neuromodulation/Parkinson_PSM/PPMI'
    csv_path  = '/home/georg-tirpitz/Documents/Neuromodulation/ddbm/out/MOCA/level2/moca_stim.csv'
    std_map   = '/home/georg-tirpitz/Documents/PD-PropensityMatching/covariate_names.csv'

    # Provide mapping ONCE via constructor or first load_ppmi call
    ppmi_data = Data(path_ppmi, foldertype="PPMI", covariate_names=std_map)

    # DBS
    ppmi_noU = ppmi_data.match_dbs(csv_path, quest="moca", STN=True, use_updrs=False)
    ppmi_U   = ppmi_data.match_dbs(csv_path, quest="moca", STN=True, use_updrs=True)

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

    # Example: lost PATNOs
    lost_patnos = sorted(set(ppmi_noU["PATNO"].unique()) - set(ppmi_U["PATNO"].unique()))
    out_csv = "lost_patnos_when_using_updrs.csv"
    pd.Series(lost_patnos, name="PATNO").to_csv(out_csv, index=False)
    print(f"Saved {len(lost_patnos)} lost PATNOs to: {out_csv}")
