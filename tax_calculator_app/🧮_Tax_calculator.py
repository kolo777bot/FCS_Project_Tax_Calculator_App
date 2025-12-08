#Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
#import loaders.load_datasets as datasets
#import deductions.mandatory_deductions as md
#import deductions.optional_deductions as od
#import tax_calculations.total_income_tax as t

##########################################################################################
#Front end with Streamlit
#Sidebar
st.set_page_config(page_title="Tax Calcualator", page_icon="🧮")
st.sidebar.success("Welcome to the St. Gallen tax calculator!")
#st.sidebar.header("Tax calculator")

#Title and infobox
st.title("🧮 St. Gallen Tax Calculator 2025")

st.info("With this app you can calculate your income tax and find out where you have the potential of saving money by finding unused tax saving options!")

#Input widgets for relevant user data
with st.container(border=False):
    #Title
    st.write("### Input your personal data here:")
    #Layout
    col1, col2,col3 = st.columns(3)
    ##Input widgets
    #First column
    marital_status = col1.selectbox("What is your martial status?", 
                                    ("Single", "Married"), 
                                    index=None, placeholder = "Select...")
    
    age = col1.number_input("What is your age?", 
                            min_value=0, max_value=117, 
                            value=None, step=1, placeholder="Type...", key=1)
    
    employed = col1.selectbox("Are you employed or self-employed?", 
                              ("Employed", "Self-employed"), 
                              index=None, placeholder = "Select...")
    if employed == "Employed":
        employed = True
    else:
        employed = False
        
    income_gross = col1.number_input("Gross income 2025 in CHF", 
                               min_value=0, value=0, step=5000, 
                               placeholder="Type in your income in CHF...")
    
    taxable_assets = col1.number_input("Taxable assets in CHF", 
                                  min_value=0, value=0, step=5000, 
                                  placeholder="Type in your income in CHF...") 
    
    #Second column
    church_affiliation = col2.selectbox("What is your confession?", 
                                ("Roman Catholic", "Protestant", "Christian Catholic", "Other/None"), 
                                index=None, placeholder = "Select...")
    
    #Children select + popups
    children = col2.selectbox("Children?", ("Yes", "No"))
    if children == "Yes":
        with col2.expander("Child data"):
            number_of_children_under_7 =  col2.number_input("How many children under 7 years old?",
                                                   min_value=0,max_value=99, 
                                                   value=0, step=1, placeholder="Type...", key=2)
            number_of_children_7_and_over = col2.number_input("How many children age 7 and older?",
                                                min_value=0, max_value=99, 
                                                value=0, step=1, key=3, placeholder="Type...")  
            number_of_children = number_of_children_under_7 + number_of_children_7_and_over  


#Layout outside of the container
left,middle, right = st.columns(3)

##########################################################################################
#Tax calculation
if left.button("Calculate", type="primary"):
    taxes = 123456.7891

##########################################################################################
####Determine deductions
##Mandatory deductions
social_deductions_total = md.get_total_social_deductions(income_gross, employed)
bv_minimal_contribution = md.get_mandatory_pension_contribution(income_gross, age)
total_mandatory_deductions = md.get_total_mandatory_deductions(income_gross, age, employed)

##Optional deductions
#Optional deduction federal 
federal_optional_deductions = od.calculate_federal_optional_deductions( 
    income_gross,
    employed,
    marital_status,
    number_of_children,
    contribution_pillar_3a,
    total_insurance_expenses,
    travel_expenses_main_income,
    child_care_expenses_third_party) #Returns a dict with individual and total federal optional deductions

total_optimal_deduction_federal = federal_optional_deductions["total_federal_optional_deductions"]
    
#Optional deduction cantonal 
cantonal_optional_deduction = od.calculate_cantonal_optional_deductions(
    income_gross,
    employed,
    marital_status,
    number_of_children,
    contribution_pillar_3a,
    total_insurance_expenses,
    travel_expenses_main_income,
    child_care_expenses_third_party,
    is_two_income_couple,
    taxable_assets,
    child_education_expenses,
    number_of_children_under_7,
    number_of_children_7_and_over)

total_optional_deduction_cantonal = cantonal_optional_deduction["total_cantonal_optional_deductions"]



   
##########################################################################################
###Calculate net income for cantonal and federal tax

income_net_federal = income_gross - (total_mandatory_deductions + total_optimal_deduction_federal)
income_net_cantonal = income_gross - (total_mandatory_deductions + total_optional_deduction_cantonal)


##########################################################################################
# Calculating tax
#getting datasets
tax_rates_federal = datasets.load_federal_tax_rates()
tax_rates_cantonal = datasets.load_cantonal_base_tax_rates()
tax_multiplicators_cantonal_municipal = datasets.load_cantonal_municipal_church_multipliers()

if left.button("Calculate", type="primary"):
    income_tax_dictionary = t.calculation_total_income_tax(
        tax_rates_federal,
        tax_rates_cantonal,
        tax_multiplicators_cantonal_municipal,
         marital_status=marital_status,
         number_of_children=number_of_children,
         income_net_federal=income_net_federal,
         income_net_cantonal=income_net_cantonal,
         commune=commune,
         church_affiliation=church_affiliation,
)

##########################################################################################   
#Display results
st.metric(label="Your estimated tax in 2025", value=f"CHF {taxes:,.2f}")

print("\n===== Income Tax Result =====")
for key, value in income_tax_dictionary.items():
    print(f"{key:35} : CHF {value:,.0f}")
    

