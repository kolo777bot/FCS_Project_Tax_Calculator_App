import tax_calculations.federal_tax as fed
import tax_calculations.canton_municipal_church_tax as can
import tax_calculations.canton_base_tax as base




######################################
# Calculation total income tax
# Returns a dictionary of total tax and individual parts
######################################

def calculation_total_income_tax(
    tax_rates_federal,
    tax_rates_cantonal,
    tax_multiplicators_cantonal_municipal,
    marital_status,
    number_of_children,
    income_net,
    commune,
    church_affiliation):
    
    # federal tax
    federal_tax = fed.calculation_income_tax_federal(tax_rates_federal, marital_status=marital_status, number_of_children=number_of_children, income_net=income_net)

    # base cantonal tax (before multipliers)
    base_income_tax_cantonal = base.calculation_income_tax_base_SG(tax_rates_cantonal, income_net)

    # cantonal + municipal + church tax (after multipliers)
    (total_canton_municipal_church, tax_canton, tax_commune, tax_church) = can.calculation_cantonal_municipal_church_tax(
        tax_multiplicators_cantonal_municipal,
        base_income_tax_cantonal,
        commune,
        church_affiliation)

    total_income_tax = federal_tax + total_canton_municipal_church

    return {
        "federal_tax": federal_tax,
        "cantonal_base_tax": base_income_tax_cantonal,
        "cantonal_tax": tax_canton,
        "municipal_tax": tax_commune,
        "church_tax": tax_church,
        "total_cantonal_municipal_church_tax": total_canton_municipal_church,
        "total_income_tax": total_income_tax,
    }
