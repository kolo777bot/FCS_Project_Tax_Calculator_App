import streamlit as st
import pandas as pd
import numpy as np

#Sidebar
st.set_page_config(page_title="Tax calcualator", page_icon="🧮")
st.sidebar.success("Welcome to your tax calculator!")
#st.sidebar.header("Tax calculator")

#Title and infobox
st.title("🧮 Tax Calculator 2025")

st.info("With this app you can calculate your wealth and income tax and find out where you have the potential of saving money by finding unused tax saving options!")

#Input widgets for relevant user data
with st.container(border=False):
    #Title
    st.write("### Input your personal data here:")
    #Layout
    col1, col2,col3 = st.columns(3)
    #input widgets
    kanton = col1.selectbox("Please select the canton you live in.",
                            ( "Aargau", "Appenzell Ausserrhoden", "Appenzell Inerrhoden", "Basel-Land", "Basel-Stadt", "Bern", "Freiburg"), 
                            index=None, placeholder = "Select...",)
    marital_status = col1.selectbox("What is your martial status?", 
                                    ("Single", "Married", "Civil Union", "Concubinage"), 
                                    index=None, placeholder = "Select...")
    age = col1.number_input("What is your age?", 
                            min_value=0, max_value=117, 
                            value=None, step=1, placeholder="Type...", key=1)
    confession = col2.selectbox("What is your confession?", 
                                ("Roman Catholic", "Protestant", "Christian Catholic", "Other/None"), 
                                index=None, placeholder = "Select...")
    #Children select + popups
    children = col2.selectbox("Children?", ("Yes", "No"))
    if children == "Yes":
        with col2.expander("Child data"):
           number_of_children =  col2.number_input("How many?",
                                                   min_value=0,max_value=99, 
                                                   value=1, step=1, placeholder=1, key=2)
        for i in range(0,number_of_children):
            age_of_children = col2.number_input("What is the age of your child?",
                                                min_value=0, max_value=99, 
                                                value=None, step=1, key=f"kids_{i}", placeholder="Type...")
    income = col1.number_input("Gross income 2025 in CHF", 
                               min_value=0, value=0, step=5000, 
                               placeholder="Type in your income in CHF...")
    net_worth = col1.number_input("Net worth in CHF", 
                                  min_value=0, value=0, step=5000, 
                                  placeholder="Type in your income in CHF...")         

#Layout outside of the container
left,middle, right = st.columns(3)
#Tax calculation
taxes = 0
if left.button("Calculate", type="primary"):
    taxes = 123456.7891
    
    
#Display results
st.metric(label="Your estimated tax in 2025", value=f"CHF {taxes:,.2f}")
    

