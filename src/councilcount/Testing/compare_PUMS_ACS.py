import os
import pandas as pd
import numpy as np
from importlib.resources import files
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data"
#DATA_PATH = files("councilcount").joinpath("data")

# This file checks whether PUMS-backed estimates fall within the 90% MOEs of ACS-backed estimates.
# It also compares estimate-to-MOE ratios and CVs between PUMS and ACS estimates.

# Create a dictionary of PUMS variables and their ACS equivalents for each year
pums_to_acs_20 = {
    "total_households_": "DP02_0001", 
    "R181_": "DP02_0014", "R651_": "DP02_0015",
    "SCHG1_": "DP02_0054", "SCHG2_": "DP02_0055", "SCHGGrades 1-8_": "DP02_0056", "SCHGGrades 9-12_": "DP02_0057", "SCHGCollege or graduate school_": "DP02_0058",
    "SCHLLess than 9th grade_": "DP02_0060", "SCHL9th to 12th grade, no diploma_": "DP02_0061", "SCHLHigh school graduate_": "DP02_0062", "SCHLSome college, no degree_": "DP02_0063", 
    "SCHLAssociate's degree_": "DP02_0064", "SCHLBachelor's degree_": "DP02_0065", "SCHLGraduate or professional degree_": "DP02_0066",
    "MIL2_": "DP02_0070",
    "DIS1_": "DP02_0072",
    "AGE_CATUnder 18_": "DP05_0019", "AGE_CAT65 and Over_": "DP05_0024",
    "DIS_AGEWith a disability, under 18 years_": "DP02_0074", "DIS_AGEWith a disability, 18-64 years_": "DP02_0076", "DIS_AGEWith a disability, 65 years and over_": "DP02_0078",
    "MIG1_": "DP02_0080", "MIG3_": "DP02_0081", "MIG2_": "DP02_0087",
    "NATIVITY1_": "DP02_0089", "NATIVITY2_": "DP02_0094",
    "CIT1_": "DP02_0090", "CIT2_": "DP02_0093", "CIT4_": "DP02_0096", "CIT5_": "DP02_0097",
    "POBPNew York_": "DP02_0091", "POBPDiff State_": "DP02_0092", "WAOBEUROPE_": "DP02_0106", "WAOBASIA_": "DP02_0107", "WAOBAFRICA_": "DP02_0108", 
    "POBPOCEANIA_": "DP02_0109", "WAOBLATIN AMERICA_": "DP02_0110", "WAOBNORTHERN AMERICA_": "DP02_0111",
    "ENG2_": "DP02_0115",
    "COMPUTER1_": "DP02_0153", "BBAND1_": "DP02_0154", "TEL2_": "DP04_0075",
    "LBR_FRC1_": "DP03_0002", "ESR1_": "DP03_0004", "ESR3_": "DP03_0005", "ESR4_": "DP03_0006", "LBR_FRC2_": "DP03_0007", "UNEMP_": "DP03_0009",
    "JWRIP1_": "DP03_0019", "JWRIP2_": "DP03_0020", "JWTRNS2_": "DP03_0021", "JWTRNS10_": "DP03_0022", "JWTRNS11_": "DP03_0024", 
    "NAICSPAgriculture, forestry, fishing and hunting, and mining_": "DP03_0033", "NAICSPConstruction_": "DP03_0034", "NAICSPManufacturing_": "DP03_0035", "NAICSPWholesale trade_": "DP03_0036", 
    "NAICSPRetail trade_": "DP03_0037", "NAICSPTransportation and warehousing, and utilities_": "DP03_0038", "NAICSPInformation_": "DP03_0039", 
    "NAICSPFinance and insurance, and real estate and rental and leasing_": "DP03_0040", "NAICSPProfessional, scientific, and management, and administrative and waste management services_": "DP03_0041", 
    "NAICSPEducational services, and health care and social assistance_": "DP03_0042", "NAICSPArts, entertainment, and recreation, and accommodation and food services_": "DP03_0043", 
    "NAICSPOther services, except public administration_": "DP03_0044", "NAICSPPublic Administration_": "DP03_0045",
    "HINCP< $10,000_": "DP03_0052", "HINCP$10,000-$14,999_": "DP03_0053", "HINCP$15,000-$24,999_": "DP03_0054", "HINCP$25,000-$34,999_": "DP03_0055", "HINCP$35,000-$49,999_": "DP03_0056", 
    "HINCP$50,000-$74,999_": "DP03_0057", "HINCP$75,000-$99,999_": "DP03_0058", "HINCP$100,000-$149,999_": "DP03_0059", "HINCP$150,000-$199,999_": "DP03_0060", "HINCP>= $200,000_": "DP03_0061",
    "PERNP_hh1_": "DP03_0064", "SSP_hh1_": "DP03_0066", "RETP_hh1_": "DP03_0068", "SSIP_hh1_": "DP03_0070", "PAP_hh1_": "DP03_0072", "FS1_": "DP03_0074",
    "HICOV1_": "DP03_0096", "PRIVCOV1_": "DP03_0097", "PUBCOV1_": "DP03_0098", "HICOV2_": "DP03_0099",
    "TEN1_": "DP04_0110", "TEN3_": "DP04_0136",
    "OCPIP<20.0%_": "DP04_0111", "OCPIP20.0-24.9%_": "DP04_0112", "OCPIP25.0-29.9%_": "DP04_0113", "OCPIP30.0-34.9%_": "DP04_0114", "OCPIP35% or more_": "DP04_0115",
    "GRPIP<15.0%_": "DP04_0137", "GRPIP15.0-19.9%_": "DP04_0138", "GRPIP20.0-24.9%_": "DP04_0139", "GRPIP25.0-29.9%_": "DP04_0140", "GRPIP30.0-34.9%_": "DP04_0141", "GRPIP35% or more_": "DP04_0142",
    "SEX1_": "DP05_0002", "SEX2_": "DP05_0003",
    "AGEPUnder 5 years_": "DP05_0005", "AGEP5-9 years_": "DP05_0006", "AGEP10-14 years_": "DP05_0007", "AGEP15-19 years_": "DP05_0008", "AGEP20-24 years_": "DP05_0009", 
    "AGEP25-34 years_": "DP05_0010", "AGEP35 to 44 years_": "DP05_0011", "AGEP45 to 54 years_": "DP05_0012", "AGEP55 to 59 years_": "DP05_0013", "AGEP60 to 64 years_": "DP05_0014", 
    "AGEP65 to 74 years_": "DP05_0015", "AGEP75 to 84 years_": "DP05_0016", "AGEP85 years and over_": "DP05_0017", "AGE_U1818 and Over_": "DP05_0021",
    "RAC1P9_": "DP05_0035", "RAC1P1_": "DP05_0037", "RAC1P2_": "DP05_0038", "RAC1P3_": "DP05_0039", "RAC1P6_": "DP05_0047", "RAC1P7_": "DP05_0055", "RAC1P8_": "DP05_0060",
    "HISP1_": "DP05_0076", "HISP2_": "DP05_0081", "RACE_HISPWhite alone, not Hispanic or Latino_": "DP05_0082", "RACE_HISPBlack or African American alone, not Hispanic or Latino_": "DP05_0083", 
    "RACE_HISPAmerican Indian or Alaska Native alone, not Hispanic or Latino_": "DP05_0084", "RACE_HISPAsian alone, not Hispanic or Latino_": "DP05_0085", 
    "RACE_HISPNative Hawaiian and Other Pacific Islander alone, not Hispanic or Latino_": "DP05_0086", "RACE_HISPSome Other Race alone, not Hispanic or Latino_": "DP05_0087", 
    "RACE_HISPTwo or More Races, not Hispanic or Latino_": "DP05_0088",
    
    "total_pop_": "B01001_001",
    "DECADE8_": "B05005_002", "DECADE7_": "B05005_007", "DECADE6_": "B05005_012", "DECADE5_": "B05005_017",
    "MAR5_": "B06008_002", "MAR1_": "B06008_003", "MAR3_": "B06008_004", "MAR4_": "B06008_005", "MAR2_": "B06008_006", 
    "POVPIPBelow 100 percent_": "B06012_002", "POVPIP100 to 149 percent_": "B06012_003", "POVPIPAt or above 150 percent_": "B06012_004",
    "VEH0_": "B08201_002", "VEH1_": "B08201_003", "VEH2_": "B08201_004", "VEH3_": "B08201_005", "VEH4+_": "B08201_006",
    "NP1_": "B08201_007", "NP2_": "B08201_013", "NP3_": "B08201_019", "NP4+_": "B08201_025",
    "HFL1_": "B25040_002", "HFL2_": "B25040_003", "HFL3_": "B25040_004", "HFL4_": "B25040_005", "HFL5_": "B25040_006", "HFL6_": "B25040_007", "HFL7_": "B25040_008", 
    "HFL8_": "B25040_009", "HFL9_": "B25040_010",
    "LANX2_": "C16001_002",
    "LANPSpanish_": "C16001_003", "LANPFrench/Haitian/Cajun_": "C16001_006", "LANPGerman/West Germanic_": "C16001_009", "LANPRussian/Polish/Slavic_": "C16001_012", 
    "LANPOther Indo-European_": "C16001_015", "LANPKorean_": "C16001_018", "LANPChinese, incl. Mandarin, Cantonese_": "C16001_021", "LANPVietnamese_": "C16001_024", 
    "LANPTagalog, incl. Filipino_": "C16001_027", "LANPOther Asian and Pacific Island_": "C16001_030", "LANPArabic_": "C16001_033", "LANPOther and Unspecified Languages_": "C16001_036"
}


pums_to_acs_16 = {
    "total_households_": "DP02_0001", 
    "R181_": "DP02_0013", "R651_": "DP02_0014",
    "SCHG1_": "DP02_0053", "SCHG2_": "DP02_0054", "SCHGGrades 1-8_": "DP02_0055", "SCHGGrades 9-12_": "DP02_0056", "SCHGCollege or graduate school_": "DP02_0057",
    "SCHLLess than 9th grade_": "DP02_0059", "SCHL9th to 12th grade, no diploma_": "DP02_0060", "SCHLHigh school graduate_": "DP02_0061", "SCHLSome college, no degree_": "DP02_0062", 
    "SCHLAssociate's degree_": "DP02_0063", "SCHLBachelor's degree_": "DP02_0064", "SCHLGraduate or professional degree_": "DP02_0065",
    "MIL2_": "DP02_0069",
    "DIS1_": "DP02_0071",
    "AGE_CAT65 and Over_": "DP05_0021",
    "DIS_AGEWith a disability, under 18 years_": "DP02_0073", "DIS_AGEWith a disability, 18-64 years_": "DP02_0075", "DIS_AGEWith a disability, 65 years and over_": "DP02_0077",
    "MIG1_": "DP02_0079", "MIG3_": "DP02_0080", "MIG2_": "DP02_0085",
    "NATIVITY1_": "DP02_0087", "NATIVITY2_": "DP02_0092",
    "CIT1_": "DP02_0088", "CIT2_": "DP02_0091", "CIT4_": "DP02_0094", "CIT5_": "DP02_0095",
    "POBPNew York_": "DP02_0089", "POBPDiff State_": "DP02_0090", "WAOBEUROPE_": "DP02_0104", "WAOBASIA_": "DP02_0105", "WAOBAFRICA_": "DP02_0106", 
    "POBPOCEANIA_": "DP02_0107", "WAOBLATIN AMERICA_": "DP02_0108", "WAOBNORTHERN AMERICA_": "DP02_0109",
    "ENG2_": "DP02_0113",
    "TEL2_": "DP04_0075",
    "LBR_FRC1_": "DP03_0002", "ESR1_": "DP03_0004", "ESR3_": "DP03_0005", "ESR4_": "DP03_0006", "LBR_FRC2_": "DP03_0007", "UNEMP_": "DP03_0009",
    "JWRIP1_": "DP03_0019", "JWRIP2_": "DP03_0020", "JWTR2_": "DP03_0021", "JWTR10_": "DP03_0022", "JWTR11_": "DP03_0024", 
    "NAICSPAgriculture, forestry, fishing and hunting, and mining_": "DP03_0033", "NAICSPConstruction_": "DP03_0034", "NAICSPManufacturing_": "DP03_0035", "NAICSPWholesale trade_": "DP03_0036", 
    "NAICSPRetail trade_": "DP03_0037", "NAICSPTransportation and warehousing, and utilities_": "DP03_0038", "NAICSPInformation_": "DP03_0039", 
    "NAICSPFinance and insurance, and real estate and rental and leasing_": "DP03_0040", "NAICSPProfessional, scientific, and management, and administrative and waste management services_": "DP03_0041", 
    "NAICSPEducational services, and health care and social assistance_": "DP03_0042", "NAICSPArts, entertainment, and recreation, and accommodation and food services_": "DP03_0043", 
    "NAICSPOther services, except public administration_": "DP03_0044", "NAICSPPublic Administration_": "DP03_0045",
    "HINCP< $10,000_": "DP03_0052", "HINCP$10,000-$14,999_": "DP03_0053", "HINCP$15,000-$24,999_": "DP03_0054", "HINCP$25,000-$34,999_": "DP03_0055", "HINCP$35,000-$49,999_": "DP03_0056", 
    "HINCP$50,000-$74,999_": "DP03_0057", "HINCP$75,000-$99,999_": "DP03_0058", "HINCP$100,000-$149,999_": "DP03_0059", "HINCP$150,000-$199,999_": "DP03_0060", "HINCP>= $200,000_": "DP03_0061",
    "PERNP_hh1_": "DP03_0064", "SSP_hh1_": "DP03_0066", "RETP_hh1_": "DP03_0068", "SSIP_hh1_": "DP03_0070", "PAP_hh1_": "DP03_0072", "FS1_": "DP03_0074",
    "HICOV1_": "DP03_0096", "PRIVCOV1_": "DP03_0097", "PUBCOV1_": "DP03_0098", "HICOV2_": "DP03_0099",
    "TEN1_": "DP04_0110", "TEN3_": "DP04_0136",
    "OCPIP<20.0%_": "DP04_0111", "OCPIP20.0-24.9%_": "DP04_0112", "OCPIP25.0-29.9%_": "DP04_0113", "OCPIP30.0-34.9%_": "DP04_0114", "OCPIP35% or more_": "DP04_0115",
    "GRPIP<15.0%_": "DP04_0137", "GRPIP15.0-19.9%_": "DP04_0138", "GRPIP20.0-24.9%_": "DP04_0139", "GRPIP25.0-29.9%_": "DP04_0140", "GRPIP30.0-34.9%_": "DP04_0141", "GRPIP35% or more_": "DP04_0142",
    "SEX1_": "DP05_0002", "SEX2_": "DP05_0003",
    "AGEPUnder 5 years_": "DP05_0004", "AGEP5-9 years_": "DP05_0005", "AGEP10-14 years_": "DP05_0006", "AGEP15-19 years_": "DP05_0007", "AGEP20-24 years_": "DP05_0008", 
    "AGEP25-34 years_": "DP05_0009", "AGEP35 to 44 years_": "DP05_0010", "AGEP45 to 54 years_": "DP05_0011", "AGEP55 to 59 years_": "DP05_0012", "AGEP60 to 64 years_": "DP05_0013", 
    "AGEP65 to 74 years_": "DP05_0014", "AGEP75 to 84 years_": "DP05_0015", "AGEP85 years and over_": "DP05_0016", "AGE_U1818 and Over_": "DP05_0018",
    "RAC1P9_": "DP05_0030", "RAC1P1_": "DP05_0032", "RAC1P2_": "DP05_0033", "RAC1P3_": "DP05_0034", "RAC1P6_": "DP05_0039", "RAC1P7_": "DP05_0047", "RAC1P8_": "DP05_0052",
    "HISP1_": "DP05_0066", "HISP2_": "DP05_0071", "RACE_HISPWhite alone, not Hispanic or Latino_": "DP05_0072", "RACE_HISPBlack or African American alone, not Hispanic or Latino_": "DP05_0073", 
    "RACE_HISPAmerican Indian or Alaska Native alone, not Hispanic or Latino_": "DP05_0074", "RACE_HISPAsian alone, not Hispanic or Latino_": "DP05_0075", 
    "RACE_HISPNative Hawaiian and Other Pacific Islander alone, not Hispanic or Latino_": "DP05_0076", "RACE_HISPSome Other Race alone, not Hispanic or Latino_": "DP05_0077", 
    "RACE_HISPTwo or More Races, not Hispanic or Latino_": "DP05_0078",
    
    "total_pop_": "B01001_001",
    "DECADE8_": "B05005_002", "DECADE7_": "B05005_007", "DECADE6_": "B05005_012", "DECADE5_": "B05005_017",
    "MAR5_": "B06008_002", "MAR1_": "B06008_003", "MAR3_": "B06008_004", "MAR4_": "B06008_005", "MAR2_": "B06008_006", 
    "POVPIPBelow 100 percent_": "B06012_002", "POVPIP100 to 149 percent_": "B06012_003", "POVPIPAt or above 150 percent_": "B06012_004",
    "VEH0_": "B08201_002", "VEH1_": "B08201_003", "VEH2_": "B08201_004", "VEH3_": "B08201_005", "VEH4+_": "B08201_006",
    "NP1_": "B08201_007", "NP2_": "B08201_013", "NP3_": "B08201_019", "NP4+_": "B08201_025",
    "HFL1_": "B25040_002", "HFL2_": "B25040_003", "HFL3_": "B25040_004", "HFL4_": "B25040_005", "HFL5_": "B25040_006", "HFL6_": "B25040_007", "HFL7_": "B25040_008", 
    "HFL8_": "B25040_009", "HFL9_": "B25040_010",
    "LANX2_": "C16001_002",
    "LANPSpanish_": "C16001_003", "LANPFrench/Haitian/Cajun_": "C16001_006", "LANPGerman/West Germanic_": "C16001_009", "LANPRussian/Polish/Slavic_": "C16001_012", 
    "LANPOther Indo-European_": "C16001_015", "LANPKorean_": "C16001_018", "LANPChinese, incl. Mandarin, Cantonese_": "C16001_021", "LANPVietnamese_": "C16001_024", 
    "LANPTagalog, incl. Filipino_": "C16001_027", "LANPOther Asian and Pacific Island_": "C16001_030", "LANPArabic_": "C16001_033", "LANPOther and Unspecified Languages_": "C16001_036"
}


pums_to_acs_11 = {
    "total_households_": "DP02_0001", 
    "R181_": "DP02_0013", "R651_": "DP02_0014",
    "SCHG1_": "DP02_0053", "SCHG2_": "DP02_0054", "SCHGGrades 1-8_": "DP02_0055", "SCHGGrades 9-12_": "DP02_0056", "SCHGCollege or graduate school_": "DP02_0057",
    "SCHLLess than 9th grade_": "DP02_0059", "SCHL9th to 12th grade, no diploma_": "DP02_0060", "SCHLHigh school graduate_": "DP02_0061", "SCHLSome college, no degree_": "DP02_0062", 
    "SCHLAssociate's degree_": "DP02_0063", "SCHLBachelor's degree_": "DP02_0064", "SCHLGraduate or professional degree_": "DP02_0065",
    "MIL2_": "DP02_0069",
    "AGE_CAT65 and Over_": "DP05_0021",
    "MIG1_": "DP02_0079", "MIG3_": "DP02_0080", "MIG2_": "DP02_0085",
    "NATIVITY1_": "DP02_0087", "NATIVITY2_": "DP02_0092",
    "CIT1_": "DP02_0088", "CIT2_": "DP02_0091", "CIT4_": "DP02_0094", "CIT5_": "DP02_0095",
    "POBPNew York_": "DP02_0089", "POBPDiff State_": "DP02_0090", "WAOBEUROPE_": "DP02_0104", "WAOBASIA_": "DP02_0105", "WAOBAFRICA_": "DP02_0106", 
    "POBPOCEANIA_": "DP02_0107", "WAOBLATIN AMERICA_": "DP02_0108", "WAOBNORTHERN AMERICA_": "DP02_0109",
    "ENG2_": "DP02_0113",
    "TEL2_": "DP04_0074",
    "LBR_FRC1_": "DP03_0002", "ESR1_": "DP03_0004", "ESR3_": "DP03_0005", "ESR4_": "DP03_0006", "LBR_FRC2_": "DP03_0007", "UNEMP_": "DP03_0009",
    "JWRIP1_": "DP03_0019", "JWRIP2_": "DP03_0020", "JWTR2_": "DP03_0021", "JWTR10_": "DP03_0022", "JWTR11_": "DP03_0024", 
    "NAICSP07Agriculture, forestry, fishing and hunting, and mining_": "DP03_0033", "NAICSP07Construction_": "DP03_0034", "NAICSP07Manufacturing_": "DP03_0035", "NAICSP07Wholesale trade_": "DP03_0036", 
    "NAICSP07Retail trade_": "DP03_0037", "NAICSP07Transportation and warehousing, and utilities_": "DP03_0038", "NAICSP07Information_": "DP03_0039", 
    "NAICSP07Finance and insurance, and real estate and rental and leasing_": "DP03_0040", "NAICSP07Professional, scientific, and management, and administrative and waste management services_": "DP03_0041", 
    "NAICSP07Educational services, and health care and social assistance_": "DP03_0042", "NAICSP07Arts, entertainment, and recreation, and accommodation and food services_": "DP03_0043", 
    "NAICSP07Other services, except public administration_": "DP03_0044", "NAICSP07Public Administration_": "DP03_0045",
    "HINCP< $10,000_": "DP03_0052", "HINCP$10,000-$14,999_": "DP03_0053", "HINCP$15,000-$24,999_": "DP03_0054", "HINCP$25,000-$34,999_": "DP03_0055", "HINCP$35,000-$49,999_": "DP03_0056", 
    "HINCP$50,000-$74,999_": "DP03_0057", "HINCP$75,000-$99,999_": "DP03_0058", "HINCP$100,000-$149,999_": "DP03_0059", "HINCP$150,000-$199,999_": "DP03_0060", "HINCP>= $200,000_": "DP03_0061",
    "PERNP_hh1_": "DP03_0064", "SSP_hh1_": "DP03_0066", "RETP_hh1_": "DP03_0068", "SSIP_hh1_": "DP03_0070", "PAP_hh1_": "DP03_0072", "FS1_": "DP03_0074",
    "TEN1_": "DP04_0108", "TEN3_": "DP04_0134",
    "OCPIP<20.0%_": "DP04_0109", "OCPIP20.0-24.9%_": "DP04_0110", "OCPIP25.0-29.9%_": "DP04_0111", "OCPIP30.0-34.9%_": "DP04_0112", "OCPIP35% or more_": "DP04_0113",
    "GRPIP<15.0%_": "DP04_0135", "GRPIP15.0-19.9%_": "DP04_0136", "GRPIP20.0-24.9%_": "DP04_0137", "GRPIP25.0-29.9%_": "DP04_0138", "GRPIP30.0-34.9%_": "DP04_0139", "GRPIP35% or more_": "DP04_0140",
    "SEX1_": "DP05_0002", "SEX2_": "DP05_0003",
    "AGEPUnder 5 years_": "DP05_0004", "AGEP5-9 years_": "DP05_0005", "AGEP10-14 years_": "DP05_0006", "AGEP15-19 years_": "DP05_0007", "AGEP20-24 years_": "DP05_0008", 
    "AGEP25-34 years_": "DP05_0009", "AGEP35 to 44 years_": "DP05_0010", "AGEP45 to 54 years_": "DP05_0011", "AGEP55 to 59 years_": "DP05_0012", "AGEP60 to 64 years_": "DP05_0013", 
    "AGEP65 to 74 years_": "DP05_0014", "AGEP75 to 84 years_": "DP05_0015", "AGEP85 years and over_": "DP05_0016", "AGE_U1818 and Over_": "DP05_0018",
    "RAC1P9_": "DP05_0030", "RAC1P1_": "DP05_0032", "RAC1P2_": "DP05_0033", "RAC1P3_": "DP05_0034", "RAC1P6_": "DP05_0039", "RAC1P7_": "DP05_0047", "RAC1P8_": "DP05_0052",
    "HISP1_": "DP05_0066", "HISP2_": "DP05_0071", "RACE_HISPWhite alone, not Hispanic or Latino_": "DP05_0072", "RACE_HISPBlack or African American alone, not Hispanic or Latino_": "DP05_0073", 
    "RACE_HISPAmerican Indian or Alaska Native alone, not Hispanic or Latino_": "DP05_0074", "RACE_HISPAsian alone, not Hispanic or Latino_": "DP05_0075", 
    "RACE_HISPNative Hawaiian and Other Pacific Islander alone, not Hispanic or Latino_": "DP05_0076", "RACE_HISPSome Other Race alone, not Hispanic or Latino_": "DP05_0077", 
    "RACE_HISPTwo or More Races, not Hispanic or Latino_": "DP05_0078",
    
    "total_pop_": "B01001_001",
    "MAR5_": "B06008_002", "MAR1_": "B06008_003", "MAR3_": "B06008_004", "MAR4_": "B06008_005", "MAR2_": "B06008_006", 
    "POVPIPBelow 100 percent_": "B06012_002", "POVPIP100 to 149 percent_": "B06012_003", "POVPIPAt or above 150 percent_": "B06012_004",
    "VEH0_": "B08201_002", "VEH1_": "B08201_003", "VEH2_": "B08201_004", "VEH3_": "B08201_005", "VEH4+_": "B08201_006",
    "NP1_": "B08201_007", "NP2_": "B08201_013", "NP3_": "B08201_019", "NP4+_": "B08201_025",
    "HFL1_": "B25040_002", "HFL2_": "B25040_003", "HFL3_": "B25040_004", "HFL4_": "B25040_005", "HFL5_": "B25040_006", "HFL6_": "B25040_007", "HFL7_": "B25040_008", 
    "HFL8_": "B25040_009", "HFL9_": "B25040_010",
}

# List of columns in PUMS files we don't need to check (levels of variables that aren't in councilcount)
pums_skip_20 = [
    "AGE_CAT18 to 64_", "BROADBND1_", "BROADBND2_", "BBAND2_", "COMPOTHX1_", "COMPOTHX2_", "COMPUTER2_", "ENG1_", 
    "HISPEED1_", "HISPEED2_", "JWTRNS12_", "JWTRNS1_", "JWTRNS7_", "JWTRNS8_", "JWTRNS9_", "LANX1_", 
    "MIL1_", "MIL4_", "SMARTPHONE1_", "SMARTPHONE2_", "TABLET1_", "TABLET2_", "TEN2_", "TEN4_",
]
pums_skip_16 = [
    "AGE_CATUnder 18_", "AGE_CAT18 to 64_", "ENG1_", "JWTR12_", "JWTR1_", "JWTR7_", 
    "JWTR8_", "JWTR9_", "LANX1_", "MIL1_", "MIL4_", "TEN2_", "TEN4_"
]
pums_skip_11 = [
    "AGE_CATUnder 18_", "AGE_CAT18 to 64_", "DECADE5_", "DECADE6_", "DECADE7_", "ENG1_", "JWTR12_", "JWTR1_", "JWTR7_", "JWTR8_", "JWTR9_", 
    "LANPArabic_", "LANPChinese, incl. Mandarin, Cantonese_", "LANPFrench/Haitian/Cajun_", "LANPGerman/West Germanic_", "LANPKorean_", 
    "LANPN/A (less than 5 years old/speaks only English)_", "LANPOther Asian and Pacific Island_", "LANPOther Indo-European_", 
    "LANPOther and Unspecified Languages_", "LANPRussian/Polish/Slavic_", "LANPSpanish_", "LANPTagalog, incl. Filipino_", "LANPVietnamese_", 
    "LANX1_", "LANX2_", "MIL1_", "MIL5_", "TEN2_", "TEN4_",
]

# Handle unemployment rate differently
special_suffix_pairs = {
    "UNEMP_": {
        "pums_est": "_PE",
        "pums_moe": "_PM",
        "acs_est": "PE",
        "acs_moe": "PM"
    }
}

geos = ["borough", "councildist_2023", "councildist_2013", "communitydist", "schooldist", "policeprct", "nta", "modzcta"]
years = [2023, 2016, 2011]
all_violations = []

moe_summary = []

for geo in geos:
    for year in years:
        if year == 2023:
            pums_to_acs = pums_to_acs_20
            pums_skip = pums_skip_20
        elif year == 2016:
            pums_to_acs = pums_to_acs_16
            pums_skip = pums_skip_16
        else:
            pums_to_acs = pums_to_acs_11
            pums_skip = pums_skip_11

        pums_skip_tuple = tuple(pums_skip)

        acs_file = f"{DATA_PATH}/ACS Estimates/demo-estimates_by-{geo}_{year}-ACS5Year.xlsx"
        pums_file = f"{DATA_PATH}/{geo}-geographies_puma_{year}.csv"

        if not (os.path.exists(acs_file) and os.path.exists(pums_file)):
            print(f"Missing files for {geo}, {year}")
            continue
        
        print(f"\nProcessing {geo}, {year}")

        acs_df = pd.read_excel(acs_file)
        pums_df = pd.read_csv(pums_file)

        leftmost_col = pums_df.columns[0]

        # Merge once on geography ID
        merged_df = acs_df.merge(
            pums_df,
            on=leftmost_col,
            how="inner",
            suffixes=("_ACS", "_PUMS")
        )

        if merged_df.empty:
            print(f"No overlapping geographies for {geo}, {year}")
            continue

        # Determine expected PUMS estimate columns
        expected_est_cols = []
        for prefix in pums_to_acs:
            if prefix.startswith(pums_skip_tuple):
                continue

            if prefix in special_suffix_pairs:
                expected_est_cols.append(prefix + "PE")
            else:
                expected_est_cols.append(prefix + "E")

        missing_est_cols = [c for c in expected_est_cols if c not in pums_df.columns]

        for col in missing_est_cols:
            print(f"PUMS estimate column missing: {col}")

        
        pums_estimate_cols = []
        for c in pums_df.columns:
            if c.startswith(pums_skip_tuple):
                continue
            if c.endswith("_E"):
                pums_estimate_cols.append(c)
            elif c.endswith("_PE"):
                prefix = c[:-2]  # restore UNEMP_
                if prefix in special_suffix_pairs:
                    pums_estimate_cols.append(c)

        for col in pums_estimate_cols:
            if col == leftmost_col:
                continue

            if col.endswith("_PE"):
                pums_prefix = col[:-3] + "_"
                suffix_rule = special_suffix_pairs.get(pums_prefix)
                pums_moe_col = col[:-2] + "PM"
            else:
                pums_prefix = col[:-1]
                suffix_rule = None
                pums_moe_col = col[:-1] + "M"

            if pums_prefix is None:
                print(f"PUMS prefix not found for column: {col}")
                continue
            if pums_prefix in pums_to_acs:
                acs_prefix = pums_to_acs[pums_prefix]
            else:
                print(f"PUMS prefix not found for column: {pums_prefix}")
                continue

            if suffix_rule:
                acs_est_col = f"{acs_prefix}{suffix_rule["acs_est"]}"
                acs_moe_col = f"{acs_prefix}{suffix_rule["acs_moe"]}"
            else:
                acs_est_col = f"{acs_prefix}E"
                acs_moe_col = f"{acs_prefix}M"

            pums_moe_col = col[:-1] + "M"
            if pums_moe_col not in pums_df.columns:
                print(f"PUMS MOE column missing: {pums_moe_col}")
                continue
            if acs_est_col not in acs_df.columns:
                print(f"ACS estimate column missing: {acs_est_col}")
                continue
            if acs_moe_col not in acs_df.columns:
                print(f"ACS MOE column missing: {acs_moe_col}")
                continue
            
            # Vectorized calculation
            acs_est = merged_df[acs_est_col]
            acs_moe = merged_df[acs_moe_col]
            # acs_se = acs_moe / 1.645
            pums_est = merged_df[col]
            pums_moe = merged_df[pums_moe_col]
            # pums_se = pums_moe / 1.645

            # --- ACS metrics for MOE/CV Comparison ---
            acs_valid = (acs_est != 0) & (~acs_est.isna()) & (~acs_moe.isna())
            acs_ratio = (acs_moe[acs_valid] / acs_est[acs_valid]).replace([np.inf, -np.inf], np.nan)
            acs_cv_col = acs_est_col[:-1] + "V"
            acs_cv = merged_df[acs_cv_col] if acs_cv_col in merged_df else pd.Series(np.nan, index=merged_df.index)

            # --- PUMS metrics for MOE/CV Comparison ---
            pums_valid = (pums_est != 0) & (~pums_est.isna()) & (~pums_moe.isna())
            pums_ratio = (pums_moe[pums_valid] / pums_est[pums_valid]).replace([np.inf, -np.inf], np.nan)
            pums_cv_col = col[:-1] + "V"
            pums_cv = merged_df[pums_cv_col] if pums_cv_col in merged_df else pd.Series(np.nan, index=merged_df.index)

            # store row-level results
            moe_summary.append(pd.DataFrame({
                "Geo": geo,
                "Year": year,
                "Dataset": ["ACS"] * len(acs_ratio) + ["PUMS"] * len(pums_ratio),
                "Ratio": pd.concat([acs_ratio, pums_ratio]),
                "CV": pd.concat([acs_cv[acs_valid], pums_cv[pums_valid]])
            }))

            # ACS independence test: potentially shaky applicability here (meant to compare ACS stats across geographies)
            # sig_test = np.abs((acs_est - pums_est)/np.sqrt(acs_se ** 2 + pums_se ** 2))

            acs_lower = acs_est - acs_moe
            acs_upper = acs_est + acs_moe
            pums_lower = pums_est - pums_moe
            pums_upper = pums_est + pums_moe

            # Boolean mask of violations
            # outside_mask = sig_test > 1.645
            outside_mask = (pums_lower > acs_upper ) | (pums_upper < acs_lower)

            if outside_mask.any():
                violations = pd.DataFrame({
                    "Geo": geo,
                    "Year": year,
                    "Geo_ID": merged_df.loc[outside_mask, leftmost_col],
                    "PUMS_Column": col,
                    "PUMS_Estimate": pums_est[outside_mask],
                    "PUMS_MOE": pums_moe[outside_mask],
                    "ACS_Estimate": acs_est[outside_mask],
                    "ACS_MOE": acs_moe[outside_mask],
                    "Difference": pums_est[outside_mask] - acs_est[outside_mask],
                    # "Z_Statistic": sig_test[outside_mask]
                })

                all_violations.append(violations)
# Combine Results
if all_violations:
    final_report = pd.concat(all_violations, ignore_index=True)
    final_report.to_csv("Mismatched Estimates.csv", index=False)
    print("\nMismatched Estimates report saved to Mismatched Estimates.csv")
else:
    print("\nNo out-of-MOE values found.")

"""""
Notes of differences:

    Tough Harmonization (no clear PUMS equivalent to ACS):
        COMPUTER:
            COMPOTHX + TABLET + SMARTPHONE gets us 97% of the way there, but we still undercount by 6-25k per borough
    
    Differences Not to Correct (likely can't be fixed):
    - Small discrepancies: AGEP, SEX (wonder if ACS is missing the SEX+AGE thing they tied together, these also have tight MOEs), 
        HISP1 (just in Bronx, which ACS says has no MOE), RACE-HISP indigenous (caps out at 0.3% of a borough's population)
    - 2023 MIG3 is a correction for ACS, not PUMS

Upgrades:
    - Updated MAR5 to only include 15+
    - Updated SCHL to only include 25+ (?)
    - Broke down birthplace between POBP and WAOB
    - Updated TEN1/TEN3 to only include non-missing OCPIP/GRPIP and positive HINCP. (???)
    - Only harmonized OCPIP for households with TEN == 1.
        That should do it, but double-check with non-missing SMOCP if errors persist.
    - Separated the military out of your ESR estimates (DP03_0004 aligns best with ESR1 and 2, 0006 with 4 and 5)
    - Updated LBR_FRC2 to only correspond to ESR6.
    - Gave NAICSP a facelift to just count 16+ with ESR 1 or 2. 
        If that's not enough, look for ways to exclude the army.
    - Cut DIS, HICOV, PRICOV, PUBCOV to Civilian Noninstitutionalized Pop with TYPEHUGQ (2023, HH) or TYPE (2016/2011, HH) != 2 and pre-recode ESR != 4
    - Added BROADBND to help estimate DP02_0154E.
    - Added SMARTHPONE to help estimate COMPUTER
    - Added LANX to estimate DP02_0013.
    - Fixed the income/earnings discrepancy by aggregating PERNP at the household level
    - Updated HINCP by using ADJINC and counting negative income as <$10K
    - Calculated unemployment rate at the very end as unemployed / labor force. Gave it a PE, PM, and V.
"""

# MOE/CV Comparison
moe_df = pd.concat(moe_summary, ignore_index=True)

geo_summary = (moe_df.groupby(["Geo", "Dataset"])
    .agg(Avg_Ratio=("Ratio", "mean"), Pct_CV_Over_20=("CV", lambda x: (x > 20).mean() * 100))
    .reset_index())

overall_summary = (moe_df.groupby("Dataset")
    .agg(Avg_Ratio=("Ratio", "mean"), Pct_CV_Over_20=("CV", lambda x: (x > 20).mean() * 100))
    .reset_index())

overall_summary["Geo"] = "All Geos"
final_summary = pd.concat([geo_summary, overall_summary], ignore_index=True)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# MOE ratio
sns.barplot(
    data=final_summary,
    x="Geo",
    y="Avg_Ratio",
    hue="Dataset",
    ax=axes[0]
)

axes[0].set_title("Average MOE / Estimate")
axes[0].tick_params(axis='x', rotation=45)

# CV > 20
sns.barplot(
    data=final_summary,
    x="Geo",
    y="Pct_CV_Over_20",
    hue="Dataset",
    ax=axes[1]
)

axes[1].set_title("% of Estimates with CV > 20")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Check whether PUMS has systematically larger uncertainty or just more extreme outliers
sns.boxplot(data=moe_df, x="Dataset", y="Ratio")
plt.ylim(0, 1)