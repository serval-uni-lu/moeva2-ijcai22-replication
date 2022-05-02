
import numpy as np
import pandas as pd
from src.utils import in_out

config = in_out.get_parameters()


def run(
    RAW_DATA_PATH=config["common"]["paths"]["raw_data"],
    DATASET_PATH=config["common"]["paths"]["dataset"],
):
    SEPARATOR = '---------'

    # ------------------- LOADING DATA

    print(SEPARATOR)
    print('Loading data')
    start_df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    loans = start_df.copy(deep=True)

    # target

    loans = loans.loc[loans['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # Remove data with more than 30% missing

    missing_fractions = loans.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    print(drop_list)

    # Only keep features known by investors

    keep_list = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title',
                 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status',
                 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec',
                 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title',
                 'total_acc', 'verification_status', 'zip_code']
    drop_list = [col for col in loans.columns if col not in keep_list]
    print(drop_list)

    loans.drop(labels=drop_list, axis=1, inplace=True)

    # Remove id, to specific

    loans.drop('id', axis=1, inplace=True)

    # convert term to integer

    loans['term'] = loans['term'].apply(lambda s: np.int8(s.split()[0]))

    # Remove grade, redundant

    loans.drop('sub_grade', axis=1, inplace=True)

    # Remove emp_title to many different values

    loans.drop(labels='emp_title', axis=1, inplace=True)
    loans.drop('title', axis=1, inplace=True)
    loans.drop(labels=['zip_code', 'addr_state'], axis=1, inplace=True)

    # Convert emp_length

    loans['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
    loans['emp_length'].replace('< 1 year', '0 years', inplace=True)

    def emp_length_to_int(s):
        if pd.isnull(s):
            return s
        else:
            return np.int8(s.split()[0])

    loans['emp_length'] = loans['emp_length'].apply(emp_length_to_int)

    # Home home ownerish replace any/none to other

    loans['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)

    # Date

    loans['earliest_cr_line'] = pd.to_datetime(loans['earliest_cr_line'].fillna('1900-01-01')).apply(
        lambda x: int(x.strftime('%Y%m')))
    loans['earliest_cr_line'] = loans['earliest_cr_line'].replace({190001: np.nan})
    loans['issue_d'] = pd.to_datetime(loans['issue_d']).apply(lambda x: int(x.strftime('%Y%m')))

    #  fico_range_low fico_range_high are correlated, take average

    loans['fico_score'] = 0.5 * loans['fico_range_low'] + 0.5 * loans['fico_range_high']
    loans.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)

    # grade

    mapping_dict = {
        "grade": {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7
        }
    }
    loans.replace(mapping_dict, inplace=True)

    # To binary

    loans = pd.get_dummies(loans, columns=['initial_list_status', 'application_type'], drop_first=True)

    # Feature creation

    def date_feature_to_month(df, feature):
        return (df[feature] / 100).apply(np.floor) * 12 + (df[feature] % 100)

    def ratio_pub_rec_pub_rec_bankruptcies(pub_rec_bankruptcies, pub_rec):
        if pub_rec > 0:
            return pub_rec_bankruptcies / pub_rec
        else:
            return -1

    loans['ratio_loan_amnt_annual_inc'] = loans['loan_amnt'] / loans['annual_inc']
    loans['ratio_open_acc_total_acc'] = loans['open_acc'] / loans['total_acc']
    loans['diff_issue_d_earliest_cr_line'] = date_feature_to_month(loans, 'issue_d') - date_feature_to_month(loans,
                                                                                                             'earliest_cr_line')
    loans['ratio_pub_rec_diff_issue_d_earliest_cr_line'] = loans['pub_rec'] / loans['diff_issue_d_earliest_cr_line']
    loans['ratio_pub_rec_bankruptcies_diff_issue_d_earliest_cr_line'] = loans['pub_rec_bankruptcies'] / loans[
        'diff_issue_d_earliest_cr_line']
    loans['ratio_pub_rec_bankruptcies_pub_rec'] = loans.apply(
        lambda x: ratio_pub_rec_pub_rec_bankruptcies(x.pub_rec_bankruptcies, x.pub_rec), axis=1)

    # To get_dummies

    loans = pd.get_dummies(loans, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=False)

    # Convert to charge off

    loans['charged_off'] = (loans['loan_status'] == 'Charged Off').apply(np.uint8)
    loans.drop('loan_status', axis=1, inplace=True)

    loans.dropna(inplace=True)

    print(SEPARATOR)
    print('Saving dataset', loans.shape)
    print(loans.columns)
    loans.to_csv(DATASET_PATH, index=False)
    # pd.DataFrame(loans.dtypes).to_csv('venus_dtypes.csv', index=True)


if __name__ == "__main__":
    run()
