import pandas as pd
import data.constants as c
import app as a

###################################
# Pillar 1 Mandatory Deductions: Social Deductions
# Pillar 2 Minimal Contributiuons 
# Optional Deductions
###################################


def get_total_social_deductions():
    alv_total = c.alv_rate_employed * min(a.income_gross, c.alv_income_ceiling)

    #Calculating the total social deductions for employed and self-employed:
    if a.employed == True:
        social_deductions_total = a.income_gross * (c.ahv_rate_employed + c.iv_rate_employed + c.eo_rate_employed) + alv_total
    else: 
        social_deductions_total = a.income_gross * (c.ahv_rate_self_employed + c.iv_rate_self_employed + c.eo_rate_employed) 
    
    return social_deductions_total


### Determine Minimal mandatory Second Pillar deductions (Occupational pension) for employed

def get_mandatory_pension_rate(age):
    coord_salary_min = 26_460
    coord_salary_max = 90_720

    if a.income_gross < coord_salary_min or a.age < 25:
        bv_rate = 0
    elif age < 25:
        bv_rate = 0
    elif 25 <= age <= 34:
        bv_rate = 0.07 
    elif 35 <= age <= 44:
        bv_rate = 0.1
    elif 45 <= age <= 54:
        bv_rate = 0.15
    elif 55 <= age <= 65:
        bv_rate = 0.18 
    else:
        bv_rate = 0
    
    bv_minimal_contribution = bv_rate * min(a.income_gross, coord_salary_max)
    
    return bv_minimal_contribution



def get_total_mandatory_deductions():
    social_deductions_total = get_total_social_deductions()
    bv_minimal_contribution = get_mandatory_pension_rate(a.age)
    total_mandatory_deductions = social_deductions_total + bv_minimal_contribution
    
    return total_mandatory_deductions