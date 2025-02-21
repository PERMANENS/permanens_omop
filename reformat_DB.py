##########################################################################################
# PERMANENS tools
#
# OMOP data extraction and predictor variable generation tool
#
# Manuel Pastor (manuel.pastor@upf.edu)
# 2025
#
##########################################################################################

import os
import sys
import yaml
import polars as pl
import numpy as np
import argparse
import psycopg2
from datetime import datetime
from dateutil.relativedelta import relativedelta

###################################################################################
local_path = 'C:/Users/manue/Nextcloud2/Permanens/'
permanens_path = '.'
###################################################################################
DB_NAME = 'permanens_v6'
HOST = 'localhost'
PORT = 5432
USER = 'giacomo'
###################################################################################

generic_list = ["sex", "age", "events"]

def define_paths ():
    ''' definition of input and output file names and paths
    '''

    paths_dict = {}
    paths_dict ['ATC_path'] = os.path.join(permanens_path, 'mappings/ATC_curation_20241203.tsv')
    paths_dict ['Conditions_path'] = os.path.join(permanens_path, 'mappings/PER_ALL_ORI_VARS_MAPPING_IMIM_20241129.tsv')

    for ikey in paths_dict:
        if not os.path.isfile(paths_dict[ikey]):
            print (f'unable to find {paths_dict[ikey]}')
            exit(0)

    paths_dict ['data_path'] = os.path.join(local_path, 'extractions/Extraction_04-12-24/OMOP_FULL_202412191536.csv')
    paths_dict ['output'] = './matrix/matrix.tsv'
    
    return paths_dict


def load_csv (data_path):
    ''' load an OMOP extraction in CSV format
    '''
 
    visits_df = pl.scan_csv(data_path,  separator='\t', schema_overrides={'visit_start_date':pl.Date,'visit_end_date':pl.Date})

    visits_df = visits_df.select  (["visit_id", "person_id", "gender", "year_of_birth", "observation", "healthcare_concept_id", "visit_start_date", "visit_end_date"]
                    ).filter (pl.col("healthcare_concept_id")==9203
                    ).filter (pl.col("observation").is_in(['potential_cases_fin: 1','potential_cases_fin: 2', 'potential_cases_fin: 3' ])
                    ).filter (pl.col("visit_end_date").is_between (datetime(2014, 1, 1),datetime(2019, 12, 31))
                    ).unique(
                    ).collect ()  
    
    patients = visits_df.select(pl.col('person_id')).unique()
    
    minipl=pl.scan_csv(data_path,  separator='\t', schema_overrides={'condition_start_date':pl.Date}
                    ).select  (["person_id", "condition_icd_code", "condition_start_date", "observation","atc","healthcare_concept_id", "death_cause_concept_id"]
                    ).filter (pl.col("person_id").is_in(patients)
                    ).collect ()  

    return visits_df, minipl, patients

def load_DB (password):
    ''' extracts the data directly from the OMOP instance, in PostgreSQL
        the parameters for the access are defined in the hea
    '''

    #Connect to the database
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=USER,
        password=password,
        host=HOST,
        port=PORT)

    # Define the query
    query_visit="""
        SELECT distinct
	    p.person_id,
		vo.visit_occurrence_id as visit_id,
        o.observation_source_value as observation,
		p.year_of_birth, 
		gc.concept_name as gender, 		
		vo.visit_concept_id as healthcare_concept_id,
		vo.visit_start_date,
		vo.visit_end_date
        from 
            person p
            left outer join condition_occurrence co on co.person_id = p.person_id
            left outer join concept gc on p.gender_concept_id = gc.concept_id
            left outer join visit_occurrence vo on vo.person_id = p.person_id 
            left outer join concept vc on vo.visit_concept_id = vc.concept_id
            left outer join observation o on o.visit_occurrence_id = vo.visit_occurrence_id 
            left outer join death d on d.person_id = p.person_id
	    	left outer join drug_exposure de on p.person_id = de.person_id 
        where
        	EXTRACT(YEAR FROM vo.visit_start_date)- p.year_of_birth>=18 and 
            vo.visit_concept_id=9203 and
            (o.observation_source_value = 'potential_cases_fin: 1' or
            o.observation_source_value = 'potential_cases_fin: 2' or 
            o.observation_source_value = 'potential_cases_fin: 3') and
            vo.visit_end_date >= '01-01-2014' and vo.visit_end_date <= '12-31-2019';
        """

    with conn.cursor() as cur:
        cur.execute("SET SCHEMA 'cdm_permanens_trial_6';")
        conn.commit()

    #Execute the query
    visits_df= pl.read_database(query_visit, conn, infer_schema_length=10000, schema_overrides={'visit_start_date':pl.Date,'visit_end_date':pl.Date}
                                ).select (["visit_id", "person_id","gender", "year_of_birth", "observation", "healthcare_concept_id", "visit_start_date", "visit_end_date"]
                                ).unique()
    
    patients = visits_df.select(pl.col(['person_id'])).unique()
    
    print("patients: "+str(patients.shape[0]))

    patients_str = ", ".join([str(id) for id in patients["person_id"].to_list()])
    
    query_minipl = f"""
    SELECT distinct
            p.person_id,
            co.condition_source_value as condition_icd_code, 
            o.observation_source_value as observation,
            dc.concept_name as drug,dc.concept_code as ATC,
            vo.visit_concept_id as healthcare_concept_id,   
            co.condition_start_date,
            case
                when d.cause_source_value = 'SuÃ¯cidis i autolesions' then 'Suicidis i autolesions'
                else d.cause_source_value
            end
            as death_cause ,
            d.cause_concept_id as death_cause_concept_id
        from
            person p
            left outer join condition_occurrence co on co.person_id = p.person_id
            left outer join visit_occurrence vo on co.visit_occurrence_id = vo.visit_occurrence_id
            left outer join concept vc on vo.visit_concept_id = vc.concept_id
            left outer join observation o on o.visit_occurrence_id = vo.visit_occurrence_id
            left outer join death d on d.person_id = p.person_id
            left outer join drug_exposure de on p.person_id = de.person_id
            left outer join concept dc on de.drug_concept_id = dc.concept_id
        where
            EXTRACT(YEAR FROM vo.visit_start_date)- p.year_of_birth>=18 and
            p.person_id in ({patients_str});  

    """

    minipl = pl.read_database(query_minipl, conn, infer_schema_length=10000000, schema_overrides={'condition_start_date':pl.Date}
                              ).select(["person_id", "condition_icd_code", "condition_start_date", "healthcare_concept_id","observation","atc", "death_cause_concept_id"])
  
    print('visits after filtering:', minipl.shape[0])

    return visits_df, minipl, patients
    
def load_ATC (ATC_mapping):
    ''' loads a table with the ATC - drug names mapping
    '''
    
    ATC_lab = 'atc_code'
    pred_lab = 'PRED_VARS_1'
    atc_map=pl.scan_csv(ATC_mapping,  separator='\t', 
                        ).select  ([ATC_lab, pred_lab]
                        ).filter((pl.col(pred_lab)!='')
                        ).collect ()

    # the ATC list is needed for obtaining a list of predictors
    # but also to assign indexes for the matrix columns
    atc_list = atc_map[pred_lab].unique().to_list()

    # the ATC dictionary will be uses to associate predictors to every ATC code
    atc_dict = {}
    for row in atc_map.iter_rows(named=True):
        code = row[ATC_lab]
        
        if not code in atc_dict:
            atc_dict[code] = ( row[pred_lab], atc_list.index(row[pred_lab]) )
    
    return (atc_list, atc_dict)

def load_Condition (Conditions_mapping):
    ''' loads a table with the ICD - condition name mappings
    '''

    head_labels = ['ICD SYSTEM', 'ICD CODE']
    
    vars_labels = ['MDX VAR1', 'MDX VAR2', 'MDX VAR3', 'ECD VAR1', 
                   'CCS VAR1', 'CCS VAR2', 'KES VAR1', 'KES VAR2', 'KES VAR3', 
                   'PAI VAR1', 'PAI VAR2']
    
    condition_map=pl.scan_csv(Conditions_mapping,  separator='\t', encoding='utf8-lossy', null_values='',
                        ).select  (head_labels+vars_labels
                        ).unique(
                        ).collect ()

    # the condition list is needed for obtaining a list of predictors
    # but also to assign indexes for the matrix columns
    condition_list = []
    for ilabel in vars_labels:
        condition_list += condition_map[ilabel].unique().to_list()
    
    condition_list = [x for x in condition_list if x is not None]

    # the condition dictionary will be uses to associate predictors to every ICD code
    condition_dict = {}
    for icondition in condition_map.iter_rows(named=True):

        # unique condition keys are a concatenation of ICD SYSTEM + ICD CODE
        code = icondition['ICD CODE']
        sys  = icondition['ICD SYSTEM']
        if sys == 'ICD9CM' and code[0] =='D':
            code = code[1:]
        
        condition_key = sys+code

        # create dictionary contents
        if not condition_key in condition_dict:
            # create a list of predictor variables associated to each ICD code
            condition_dict[condition_key] = []
            
            # iterate for every family of conditions (a column in the input table)
            for ilabel in vars_labels:
                icondition_item = icondition[ilabel]  # this is a single predictor variable name
                if icondition_item!=None:
                    condition_dict[condition_key].append( ( icondition_item, condition_list.index(icondition_item) ))

    return (condition_list, condition_dict)

def mapped_conditions(code, condition_dict):
    ''' returns the list of conditions associated to the input ICD code
    '''
    ICM = { 'CIM9MC':'ICD9CM',
            'CIM10MC': 'ICD10CM',
            'CIM10': 'ICD10'}
    
    if code is None:
        return None
    
    if code == 'No matching concept':
        return None
    
    if code.startswith('potential_cases'):
        return None
    
    code_list = code.split('-')
    key = ICM[code_list[2]]+code_list[0].replace('.','')

    if key in condition_dict:
        return (condition_dict[key])

    return None

def mapped_drug(atc, atc_dict):
    ''' returns the drug associated to the input ATC code  
    '''
    if atc is None:
        return None
    
    if atc in atc_dict:
        return ( atc_dict[atc] ) 
    
    return (None)

def mapped_event (observation, healthcare_concept):
    ''' returns True if the observation correspond to a self-harm event and the heathcase setting is the ED
    '''
    if observation == 'potential_cases_fin: 1' or observation == 'potential_cases_fin: 2' or observation == 'potential_cases_fin: 3':
        if str(healthcare_concept) == '9203':
            return True 

    return False

def positive_condition (observation, death):
    ''' returns True if the observation meets certain requirements defining a positive outcome
    '''
    if observation == 'potential_cases_fin: 1' or observation == 'potential_cases_fin: 2' or observation == 'potential_cases_fin: 3':
        return True 
    
    if str(death) == '4303690':
        return True
    
    return None


def build_collection (minipl, conditions_dict, atc_dict):
    ''' creates a dictionary with a key for every patient, 
        the value contains a dictionary with three sets: conditions, drugs and outcomes
        We used sets 
    '''
    collection = {}

    for row in minipl.iter_rows(named=True):
        person_id = row['person_id'] 

        if not person_id in collection:
            collection [person_id] = {'conditions':set(), 'drugs':set(), 'outcome': set(), 'events': set() }

        collection_item = collection[person_id]
        
        date = row['condition_start_date']

        # append conditions
        conditions = mapped_conditions ( row['condition_icd_code'], conditions_dict)
        if conditions != None and len(conditions)>0:
            for icondition in conditions:
                collection_item['conditions'].add( (date, icondition) )

        # append observations
        observations = mapped_conditions ( row['observation'], conditions_dict)
        if observations != None and len(observations)>0:
            for iobservation in observations:
                collection_item['conditions'].add( (date, iobservation) )

        # append drugs
        drug = mapped_drug (row['atc'], atc_dict)
        if drug != None:
            collection_item['drugs'].add( (date, drug) )

        # increase previous events
        if mapped_event (row['observation'], row['healthcare_concept_id']):
            collection_item['events'].add (date)

        # evaluate potential cases fin
        outcome = positive_condition(row['observation'], row['death_cause_concept_id'])
        if outcome != None:
            collection_item['outcome'].add( (date, outcome) )
    
    return collection

def build_matrix (collection, visits_df, condition_list, atc_list, ):
    ''' builds the X-Y matrix to be used by the machine-learning algorithm
        X: observations are in rows and predictors in columns, with a header 
        Y: last column, as 0/1 
    '''
    # create numpy arrays for ATC, conditions and outcomes
    num_visits = len(visits_df)
    num_conditions = len(condition_list)
    num_atc = len(atc_list)
    num_generic = len(generic_list) # gender + age + events

    generic_matrix = np.zeros ( (num_visits, num_generic), dtype="int64")
    atc_matrix = np.zeros( (num_visits, num_atc), dtype="int16")
    condition_matrix = np.zeros((num_visits, num_conditions), dtype="int16")
    outcomes_matrix = np.zeros((num_visits, 1), dtype="int16")

    count = 0
    for row in visits_df.iter_rows(named=True):
        person_id = row['person_id']
        startdate = row['visit_start_date']
        enddate = row['visit_end_date']
        
        # GENERIC | sex
        if row['gender'] == 'MALE':
            generic_matrix [count, 0] = 1
        
        # GENERIC | age
        generic_matrix [count, 1] = startdate.year - row['year_of_birth']  

        if person_id in collection:
            person_events = collection[person_id]

            #####################################################################################
            # ATCs and CONDITIONS are assigned if their start_date is within a time window of 1 YEAR 
            # before the visit startdate  
            # 
            # This window includes the visit day as well, which means that the ATCs and CONDITIONS
            # of the current visit are also included
            #####################################################################################
            # ATCS
            for iprescription in person_events['drugs']:
                idate = iprescription [0]
                idrug = iprescription [1]
                if (idate <= startdate) and (idate + relativedelta (years=1) > startdate ):
                    atc_matrix [count, idrug[1]] = 1
                
            # CONDITIONS
            for idiagnostic in person_events['conditions']:
                idate = idiagnostic [0]
                icode = idiagnostic [1]
                if (idate <= startdate) and (idate + relativedelta (years=1) > startdate):
                    condition_matrix [count, icode[1]] = 1
            
            # GENERIC | events
            for idate in person_events['events']:
                if (idate < startdate) and (idate + relativedelta (years=1) > startdate):
                    generic_matrix [count, 2] += 1

            #####################################################################################
            # OUTCOMES are assigned if their start_date is witin a time window of 1 YEAR 
            # after the visit end_date PLUS 1 day
            # 
            # The date of the visit is NOT included, which means that outcomes linked to
            # the current visit are not considered
            #####################################################################################
            # OUTCOMES
            for ioutcome in person_events['outcome']:
                idate = ioutcome [0]
                if (idate > enddate + relativedelta (days=1)) and (idate < enddate + relativedelta (years=1) ):
                    outcomes_matrix [count, 0] = 1   # outcomes only collect positives

        count +=1

    # print('generic: ', generic_matrix.sum(axis=0))
    # print('atc: ', sorted([int(j) for j in atc_matrix.sum(axis=0)]))
    # print('condition: ',sorted([int (j) for j in condition_matrix.sum(axis=0)]))
    # print('outcomes: ', outcomes_matrix.sum(axis=0))

    final_matrix = pl.from_numpy(generic_matrix, schema=generic_list, orient="row")
    final_matrix = final_matrix.hstack( pl.from_numpy(atc_matrix, schema=atc_list, orient="row"))
    final_matrix = final_matrix.hstack( pl.from_numpy(condition_matrix, schema=condition_list, orient="row"))
    final_matrix = final_matrix.hstack( pl.from_numpy(outcomes_matrix, schema=["outcome"], orient="row"))

    return (final_matrix)

def main ():

    paths_dict = define_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--password', type=str, required=False, help='Password for database access (required)')
    
    # Parse the arguments
    args = parser.parse_args()  
    if args.password != None:
        sys.stdout.write('loading input from local DB instance...\n')
        sys.stdout.flush()
        visits_df, minipl, patients = load_DB(args.password)
    else:
        sys.stdout.write('loading input csv... ')
        sys.stdout.flush()
        visits_df, minipl, patients = load_csv(paths_dict['data_path'])
        print (f'loaded {len(visits_df)} visits for {len(patients)} unique patients')

    sys.stdout.write('loading mappings... ')
    sys.stdout.flush()
    atc_list, atc_dict = load_ATC(paths_dict['ATC_path'])
    condition_list, conditions_dict = load_Condition(paths_dict['Conditions_path'])
    print (f'loaded {len(atc_list)} drugs and {len(condition_list)} conditions')
    
    with open ('./matrix/predictors.yaml','w') as handle:
        predictors_exp = {}
        predictors_exp['generic'] = generic_list
        predictors_exp['drugs'] = atc_list
        predictors_exp['conditions'] = condition_list
        yaml.dump(predictors_exp, handle)

    print ("list of cathegorized predictors exported in 'matrix/predictors.yaml'")

    sys.stdout.write('building a dictionary of patients with conditions, drugs and outcomes... ')
    sys.stdout.flush()
    collection = build_collection(minipl, conditions_dict, atc_dict)
    print ('complete')

    sys.stdout.write('building visit matrix... ')
    sys.stdout.flush()
    final_matrix = build_matrix (collection, visits_df, condition_list, atc_list)    
    shape = final_matrix.shape

    print (f'matrix has {shape[0]} visits, {shape[1]-1} predictors and 1 outcome variable')

    ofile = paths_dict['output']
    sys.stdout.write(f'writting output tsv in {ofile} ... ')
    sys.stdout.flush()
    final_matrix.write_csv (ofile, separator='\t')
    print ('success')

    print ('normal termination')

if __name__ == '__main__':
    main()
