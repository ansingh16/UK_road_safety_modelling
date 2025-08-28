# load libraries
import pandas as pd
import numpy as np
import plotly.express as px
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

import tqdm

# set working directory
data_dir = pathlib.Path('../data/') 

class UKRoadSafety:
    def __init__(self, data_dir=data_dir,years=[2023]):
        """
        Parameters
        ----------
        data_dir : pathlib.Path, default=data_dir
            Path to the directory containing the collision data.
        years : list of int, default=[2023]
            List of years for which to load the data.

        Attributes
        ----------
        df : pd.DataFrame
            Concatenated dataframe of the collision data for the given years.
            The dataframe is indexed by the 'accident_index' column.
        """
        self.data_dir = data_dir
        self.years = years
        
        # load collision data
        files = [f'{self.data_dir}/dft-road-casualty-statistics-collision-{year}.csv' for year in years]
        df_collisions = pd.concat([pd.read_csv(file, low_memory=False) for file in files])
        df_collisions.set_index('accident_index', inplace=True)
        # load vehicle data
        files = [f'{self.data_dir}/dft-road-casualty-statistics-vehicle-{year}.csv' for year in years]
        df_vehicles = pd.concat([pd.read_csv(file, low_memory=False) for file in files])
        df_vehicles.set_index('accident_index', inplace=True)
        # merge collision and vehicle data on 'accident_index'
        self.df = df_collisions.merge(df_vehicles, left_index=True, right_index=True, how='left')

