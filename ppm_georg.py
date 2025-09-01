import numpy as np
import pandas as pd 
import seaborn as sns
import sklearn as sk
from icecream import ic
from typing import Dict, Callable
import csv
import os
from datetime import datetime

# Base Data Class
class Data:
    def __init__(self, path_of_folder: str, foldertype: str = "PPMI"):
        """
        Loads the dataset from the given path and initializes the data structures.
        """
        self.path = path_of_folder
        self.foldertype = foldertype
        if foldertype=="PPMI":
            self.complete_data = self.load_ppmi(path_of_folder, foldertype)
        elif foldertype=="tuebingen":
            self.complete_data = self.load_tuebingen(path_of_folder, foldertype)
        elif foldertype=="custom":
            self.complete_data = self.load_costum(path_of_folder, foldertype)
        else:
            raise ValueError("For Folder, only 'PPMI', 'tuebingen' or 'custom' are allowed")

        self.covariates = None  # Subset of data used for matching
        self.covariates_longitude = None  # Data for longitudinal analysis
        self.df = None  # Data in format for statistical models probably numpy or something similar
        self.is_converted_to_standard = False

    def load_ppmi(self, path_of_folder: str, folder: str):
        """
        Loads in data according to PPMI layout
        """
        if folder!="PPMI":
            raise ValueError("For Folder, only 'PPMI' is allowed")
        
        datdir =path_of_folder


        stat = pd.read_csv(datdir+'/Participant_Status_13Feb2024.csv')
        dbs = pd.read_csv(datdir+'/Surgery_for_PD_Log_15Feb2024.csv')
        dbs = dbs.loc[dbs.PDSURGTP==1,:]
        medication = pd.read_csv(datdir+'/LEDD_Concomitant_Medication_Log_24Apr2024.csv')
        
        diaghist = pd.read_csv(datdir+'/PD_Diagnosis_History_21Feb2024.csv')
        diaghist.SXDT = pd.to_datetime(diaghist.SXDT)

        demo = pd.read_csv(datdir+'/Demographics_25Apr2024.csv')
        demo['SEX'] = np.array(['female', 'male'])[demo['SEX']]
        # updrs

        #Load in and merge to full updrs1
        updrs1 = pd.read_csv(datdir+'/MDS-UPDRS_Part_I_15Feb2024.csv')
        updrs1p = pd.read_csv(datdir+'/MDS-UPDRS_Part_I_Patient_Questionnaire_15Feb2024.csv')
        updrs1 = pd.merge(updrs1, updrs1p, on='PATNO', how='outer')

        #Load in and merge to full updrs2
        updrs2p = pd.read_csv(datdir+'/MDS_UPDRS_Part_II__Patient_Questionnaire_15Feb2024.csv')
        
        updrs3 = pd.read_csv(datdir+'/MDS-UPDRS_Part_III_15Feb2024.csv')


        UPDRSrig = ["NP3RIGLL",
                    "NP3RIGLU",
                    "NP3RIGN",
                    "NP3RIGRL",
                    "NP3RIGRU"]

        updrs3['rigidity'] = updrs3.loc[:,UPDRSrig].sum(1)

        updrs3["bradykinesia"] = updrs3.loc[:,"NP3BRADY"]
        UPDRSlat = ["NP3FTAP",
                    "NP3HMOV",
                    "NP3KTRM",
                    "NP3LGAG", 
                     "NP3PRSP",
                     "NP3PTRM"]

        updrs3['latindex'] = (updrs3.loc[:,[i+"R" for i in UPDRSlat]].sum(1) - updrs3.loc[:,[i+"L" for i in UPDRSlat]].sum(1))#/updrs3.loc[:,[i+"R" for i in UPDRSlat]].sum(1) + updrs3.loc[:,[i+"L" for i in UPDRSlat]].sum(1)
        latix = updrs3.groupby(["PATNO","EVENT_ID", "PDSTATE"],as_index=False).latindex.mean()
        ##average over MED ON and OFF states (only for UPDRS3)
        #updrs3 = updrs3.groupby(["PATNO","EVENT_ID"],as_index=False).NP3TOT.mean()
        updrs3 = pd.merge(updrs3,latix)

        updrs4 = pd.read_csv(datdir+'/MDS-UPDRS_Part_IV__Motor_Complications_15Feb2024.csv')
        #updrs4.NP4TOT
        mds_updrs = dict(mds_updrs1=updrs1, mds_updrs2=updrs2p, mds_updrs3=updrs3, mds_updrs4=updrs4)

        #updrs.loc[:,"UPDRS_SUMSCORE"] = updrs.loc[:,['NP1RTOT','NP1PTOT','NP2PTOT','NP3TOT','NP4TOT']].sum(1)

        moca = pd.read_csv(datdir+'/Montreal_Cognitive_Assessment__MoCA__12Mar2024.csv')

        #REM sleep questionnaire
        rbd = pd.read_csv(datdir+'/REM_Sleep_Behavior_Disorder_Questionnaire_08Feb2024.csv')
        
        ppmi_dict = {"stat":stat, "dbs":dbs, "medication": medication, "diaghist":diaghist, "demo":demo, "mds_updrs":mds_updrs, "moca":moca, "rbd":rbd}

        return ppmi_dict

    def convert_to_standard_keys(self, file: str, dbs: bool = True):
        """
        Convert the PPMI data to the standard keys, which are based on the tuebingen data format.
        """
        #Reading in the covariate names file
        if self.is_converted_to_standard:
            print("Already converted to standard")
            return
        covariate_file = file
        if not os.path.exists(covariate_file):
            raise FileNotFoundError(f"Covariate names file not found at {covariate_file}. Please provide the file.")

        covariate_dict = {}
        with open(covariate_file, mode='r') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            for rows in reader:
                key = rows[0]
                covariate_dict[key] = rows[1:]
        self.complete_data

        #Changing the keys of the dataframes to the standard keys
        #Exchange keys of UPDRS
        #UPDRS1
        self.complete_data['mds_updrs']['mds_updrs1'].columns.values[6:12] = covariate_dict['mds_updrs1'][5:11]
        self.complete_data['mds_updrs']['mds_updrs1'].columns.values[5:19] = covariate_dict['mds_updrs1'][4:18]

        #UPDRS2
        self.complete_data['mds_updrs']['mds_updrs2'].columns.values[6:20] = covariate_dict['mds_updrs2'][5:19]
        self.complete_data['mds_updrs']['mds_updrs2'].rename(columns={'INFODT': covariate_dict['mds_updrs3'][4]}, inplace=True)  
        #UPDRS3
        #questions
        self.complete_data['mds_updrs']['mds_updrs3'].columns.values[23:56] = covariate_dict['mds_updrs3'][8:41]
        #HY scale
        self.complete_data['mds_updrs']['mds_updrs3'].rename(columns={'NHY': covariate_dict['mds_updrs3'][-2]}, inplace=True)
        #total score
        self.complete_data['mds_updrs']['mds_updrs3'].rename(columns={'NP3TOT': covariate_dict['mds_updrs3'][-1]}, inplace=True)  
        #Dyskinesia presence
        self.complete_data['mds_updrs']['mds_updrs3'].rename(columns={'DYSKPRES': covariate_dict['mds_updrs3'][-4]}, inplace=True)  
        #Exam date
        self.complete_data['mds_updrs']['mds_updrs3'].rename(columns={'INFODT': covariate_dict['mds_updrs3'][4]}, inplace=True)  
        #UPDRS4    
        self.complete_data['mds_updrs']['mds_updrs4'].columns.values[[5, 9, 10, 14, 15, 16, 20]] = covariate_dict['mds_updrs4'][5:]
        # Drop specified columns from UPDRS4
        self.complete_data['mds_updrs']['mds_updrs4'].drop(self.complete_data['mds_updrs']['mds_updrs4'].columns[[6, 7, 8, 11, 12, 13, 17, 18, 19]], axis=1, inplace=True)

        #Demographics
        demo_dict = {}
        if dbs:
            dbs_and_demo = pd.merge(self.complete_data["dbs"], self.complete_data["demo"], on="PATNO", how="inner")
            demo_dict['OP_DATUM'] = pd.to_datetime(dbs_and_demo['PDSURGDT'])
            bdt = pd.to_datetime(dbs_and_demo['BIRTHDT'])
            opdt = pd.to_datetime(dbs_and_demo['PDSURGDT'])
            demo_dict['AGE_AT_OP'] = (opdt - bdt).dt.days / 365.25
            demo_dict['SEX'] = dbs_and_demo['SEX']
            demo_dict['PATNO'] = dbs_and_demo['PATNO']
        else:
            demo_dict['OP_DATUM'] = pd.to_datetime(self.complete_data['demo']['PDSURGDT'])
            demo_dict['SEX'] = self.complete_data['demo']['SEX']
            demo_dict['PATNO'] = self.complete_data['demo']['PATNO']
        #MOCA
        moca_dict = {}
        moca_dict['executive'] = self.complete_data['moca'].iloc[:, 5:9].sum(axis=1)
        moca_dict['naming'] = self.complete_data['moca'].iloc[:, 10:12].sum(axis=1)
        moca_dict['attention_numbers'] = self.complete_data['moca'].iloc[:, 13:14].sum(axis=1)
        moca_dict['attention_letters'] = self.complete_data['moca'].iloc[:, 15]
        moca_dict['attention_substract'] = self.complete_data['moca'].iloc[:, 16]
        moca_dict['language_rep'] = self.complete_data['moca'].iloc[:, 17] 
        moca_dict['language_letters'] = self.complete_data['moca'].iloc[:, 18:19].sum(axis=1)
        moca_dict['abstraction'] = self.complete_data['moca'].iloc[:, 20]
        moca_dict['reminding'] = self.complete_data['moca'].iloc[:, 21:25].sum(axis=1)
        moca_dict['orientation'] = self.complete_data['moca'].iloc[:, 26:31].sum(axis=1)
        moca_dict['total'] = self.complete_data['moca'].iloc[:, 32]  
        moca_dict = dict(zip(covariate_dict['moca'][5:16], moca_dict.values()))
        self.complete_data['moca'] = pd.concat([
            self.complete_data['moca'].iloc[:, :5],
            pd.DataFrame(moca_dict),
            self.complete_data['moca'].iloc[:, -2:]
        ], axis=1)
        self.complete_data['demo'] = pd.DataFrame(demo_dict)
        self.complete_data['moca'].rename(columns={'INFODT': covariate_dict['moca'][4]}, inplace=True)
        self.is_converted_to_standard = True
        
    
    def filter_from_csv(self, csv_path, quest, standard_key_dict_path):

        df = pd.read_csv(csv_path)
        df = df.rename(columns={col: col.replace('_pre', '') for col in df.columns if col.endswith('_pre')})
        df = df[[col for col in df.columns if not col.endswith('_post')]]

        if not self.is_converted_to_standard:
            self.convert_to_standard_keys(standard_key_dict_path, dbs=True)
        ppmi_cohort = pd.merge(self.complete_data["demo"], self.complete_data[quest], on="PATNO", how="inner")

        if quest == "moca":
            MOCA_DATA_COLS  =[
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
                "Benennen": [
                    "MoCA_Benennen"
                ],
                "Executive": [
                    "MoCA_Executive"
                ],
                "Abstraktion": [
                    "MoCA_Abstraktion"
                ],
                "Erinnerung": [
                    "MoCA_Erinnerung"
                ],
                "Orientierung": [
                    "MoCA_Orientierung"
                ]
                }
            ppmi_cohort["MoCA_sum"] = ppmi_cohort.apply(
                lambda row: row[MOCA_DATA_COLS].sum(skipna=False) if pd.isna(row["MoCA_ONLY_GES"]) else row["MoCA_ONLY_GES"],
                axis=1
                )
            pc_df = ppmi_cohort[["OP_DATUM", "TEST_DATUM","MoCA_sum"]].copy()
            for cat_name, item_cols in MOCA_CATEGORIES.items():
                pc_df[f"MoCA_{cat_name}_sum"] = ppmi_cohort[item_cols].sum(axis=1, skipna=False)

        return pc_df, df
        
    

    def load_costum(self, path_of_folder: str, foldertype: str):
        """
        Loads in data from costom (your) data folder. Has to be implemented according to your data layout
        """
        pass

    def export_covariate_names(self, full_df, path: str):
        """
        Assuming that this dataset will be the standard when it comes to dataframe keys, export column names to a csv. file.
        """
        if self.foldertype=="tuebingen":
            with open(path + '/covariate_names.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(full_df.keys())  # Write the dictionary keys as the header row

                def write_dict_to_csv(d, parent_key=''):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            write_dict_to_csv(value)
                        else:
                            writer.writerow([parent_key + key] + list(value.columns))

                write_dict_to_csv(full_df)
            
    def select_covariates(self, covariates: Dict = None):
        """
        Select a subset of data based on the specified covariates of interest. Has to be specified by the user
        """
        
        #Set covaries of interest as given in the dictionary
        if covariates is not None:
            self.covariates = covariates
        else:
            #Manually select/compute the covariates of interest
            self.covariates =  self.complete_data["mds_updrs"] # Replace with the filtered data
        self.is_converted_to_standard = True
        return self.covariates

    
    def convert_costom_to_standard_keys(self, file: str):
        """
        convert the costom data to the standard keys, which are based on the tuebingen data format
        """
        pass


    def remove_outliers(self):
        """
        Placeholder for outlier removal. To be implemented.
        """
        print("Removing outliers...")
        # Implement logic for removing outliers
        pass


# Propensity Matching Class
class PropensityMatching:
    def __init__(self, ppmi: Data, custom: Data):
        """
        Initialize the propensity matching with PPMI and DBS data.
        """
        self.ppmi = ppmi
        self.custom = custom
         
        self.ppmi_for_model = None
        self.custom_for_model = None

    def match(self, matching_method: Callable, grouping_func: Callable, classification_model: Callable):
        """patno matching...")"""
        # Implement the matching logic using the provided methods
        return None

    def match_method1(self):
        """
        Match on distance of preoperative test dates and follow up test dates.
        """
        
        def convert_dates_to_days(date_dict: Dict[str, Dict[str, list]]):
            """
            Convert dates in the format 'YYYY-MM-DD' to the number of days since the first date.
            """
            for covariate, sub_dict in date_dict.items():
                for patno, date_list in sub_dict.items():
                    # Convert string dates to datetime objects
                    date_list = [datetime.strptime(date, '%Y-%m-%d') for date in date_list if date]
                    if date_list:
                        # Find the minimum date
                        min_date = datetime(1950, 1, 1)    # Convert dates to the number of days since the minimum date
                        date_dict[covariate][patno] = [(date - min_date).days for date in date_list]
            return date_dict
        
        def find_preop_test(date_dict: Dict[str, Dict[str, list]]):
            """
            Find the last test date preceding the operation for each patient in the DBS cohort.
            """
            pass
            
        def match_on_preop():
            """
            Match patients based on the preoperative test dates.
            """
            pass

        def match_on_followup():
            """
            Match patients based on the follow-up dates.
            """
            pass
        
        
            

    def grouping_method1(self):
        """
        Example grouping method 1. Replace with actual logic.
        """
        print("Using grouping method 1...")
        pass

    def classification_model1(self):
        """
        Example classification model 1. Replace with actual logic.
        """
        print("Using classification model 1...")
        pass

        
if __name__ == "__main__":
            # Example usage
            # Set paths to your data folders
            path_ppmi = '/home/georg-tirpitz/Documents/Neuromodulation/Parkinson_PSM/PPMI'
            csv_path = '/home/georg-tirpitz/Documents/Neuromodulation/ddbm/out/MOCA/level2/moca_stim.csv'

            # Initialize Data objects
            ppmi_data = Data(path_ppmi, foldertype="PPMI")
            ppmi_cohort_df, csv_df = ppmi_data.filter_from_csv(
                csv_path, 
                quest="moca", 
                standard_key_dict_path='/home/georg-tirpitz/Documents/PD-PropensityMatching/covariate_names.csv')

            # Select covariates (customize as needed)
            

            print("Main execution completed.")
