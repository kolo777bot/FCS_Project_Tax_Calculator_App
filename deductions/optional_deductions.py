import pandas as pd
import data.constants as c
import app as a


### Define optional deductions
#Determine deduction of pillar 3a contributions
contribution_pillar_3a_threshold_employed = 7258
contribution_pillar_3a_threshold_self_employed = min(a.income_gross * 0.2, 36_288)

if a.employed == True:
    deduction_pillar_3a = min(a.contribution_pillar_3a, contribution_pillar_3a_threshold_employed)
else:
    deduction_pillar_3a = min(a.contribution_pillar_3a, contribution_pillar_3a_threshold_self_employed)

#Determine insurance premium deductions
insurance_max_deductible_amount_single = 1700
insurance_max_deductible_amount_married = 3500 

deduction_insurance_single = min(a.total_insurance_expenses, 1700)
deduction_insurance_married = min(a.total_insurance_expenses, 3500 / 2)
 #### different deductible for cantonal tax in SG. Right now ignoring this 