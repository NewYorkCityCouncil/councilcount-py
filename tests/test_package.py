# import unittest
# import pandas as pd
# import warnings
# from councilcount import estimates

# class TestYourModule(unittest.TestCase):

#     def test_dataframe_output(self):
#         # Call your function

#         acs_year = 2022
#         census_api_key = '2f42b2b59ea3d5882dd01587b2e25769f000ed56'
#         demo_dict = {
#             "DP02_0002E": "household", # Married-couple household
#             "DP02_0025E": "person", # Males 15 and over
#             "DP02_0026E": "person" # Never married males 15 and over
#             }
#         geo = "schooldist" # "councildist", "policeprct", "schooldist", "nta", "communitydist", "borough", and "city" are acceptable inputs
#         total_pop_code = "DP02_0088E" # Use this code for years 2020 and above. Use "DP02_0086E" for 2018 and earlier surveys. Use "DP02_0087E" for 2019. 
#         total_house_code = "DP02_0001E" # This code should be correct in most cases
            
#         result = estimates.generate_new_estimates(acs_year, demo_dict, geo, census_api_key, total_pop_code, total_house_code, boundary_year=None)

#         # Check if result is a DataFrame
#         self.assertIsInstance(result, pd.DataFrame)

#     def test_warning_is_raised(self):
#         # Use the warnings module to check if a warning is raised

#         acs_year = 2022
#         census_api_key = '2f42b2b59ea3d5882dd01587b2e25769f000ed56'
#         demo_dict = {
#             "DP02_0002E": "household", # Married-couple household
#             "DP02_0025E": "person", # Males 15 and over
#             "DP02_0026E": "person" # Never married males 15 and over
#             }
#         geo = "schooldist" # "councildist", "policeprct", "schooldist", "nta", "communitydist", "borough", and "city" are acceptable inputs
#         total_pop_code = "DP02_0088E" # Use this code for years 2020 and above. Use "DP02_0086E" for 2018 and earlier surveys. Use "DP02_0087E" for 2019. 
#         total_house_code = "DP02_0001E" # This code should be correct in most cases
            

#         with self.assertRaises(Warning):
#             result = estimates.generate_new_estimates(acs_year, demo_dict, geo, census_api_key, total_pop_code, total_house_code, boundary_year=None)

#     def test_warning_message(self):
#         # Capture warnings and check if the correct warning is raised
#         with self.assertRaises(Warning) as cm:
#             your_module.your_function_with_warning()

#         # Assert the message of the warning
#         self.assertEqual(str(cm.exception), "Expected warning message here")

# if __name__ == '__main__':
#     unittest.main()
