import pytest
import pandas as pd
from src.councilcount import pums_estimates

@pytest.mark.parametrize("var_code", ["SEX", "NATIVITY"])

def test_calc_proportion_moe(MOE_code):
    #given
    demo_dict = {"SEX": "person", "NATIVITY": "person"}
    variance_df = pd.read_csv("/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/test_variance_df", dtype = {"puma2020": str})
    total_pop_code = "total_popE"

    #when
    df = pums_estimates._calc_proportion_moe(demo_dict, variance_df, var_code, total_pop_code, total_house_code = None)

    #then
    assert isinstance(df, pd.DataFrame)