import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re
import math
import string
import collections
import dill

# get working directory to save files
working_dir = os.getcwd()

# Category names and ids under skincare-face; make list so dont have to parse them for every page
cat_name = ['Treatments (Face)', 'Toners', 'Cleansers'
            , 'Moisturizers', 'Masks', 'Eye Makeup Remover', 'Neck/Decollete Cream', 'Scrubs', 'Treatments (Eye)', ]
cat_id = ['705', '702', '701', '707', '703', '708', '709', '704', '706']
cat_dict = dict(zip(cat_id, cat_name))

def create_df():
    """
    join all individual dfs together
    :return: final df
    """
    df = pd.DataFrame()
    for i in range(len(cat_id)):
        df_c = dill.load(open('df_{}.pkd'.format(cat_id[i]), 'rb'))
        df = df.append(df_c)
    return df

df = create_df()

# number of reviews
df.info()

# number of unique products
len(df.pro_ids.unique())

####################pre-processing to get data in the right types#################
# ratings to float
df['ratings'] = df['ratings'].astype(float)



# remove string, then convert to decimals
df['repurchases'] = df['repurchases'].apply(lambda x: x.replace("% would repurchase", ''))
df['repurchases'] = df['repurchases'].astype(float)/100

# remove string, then convert to float
df['pkg_quals'] = df['pkg_quals'].apply(lambda x: x.replace("Package Quality: ", ''))
df['pkg_quals'] = df['pkg_quals'].astype(float)

# remove string, then convert to integers
df['prices'] = df['prices'].apply(lambda x: x.replace("Price: ", ''))
df['prices'] = df['prices'].apply(lambda x: len(x))

# split characteristics
df['chars'] = df['chars'].apply(lambda x: list(filter(None, re.split(r'\bAge:\s\b|\bSkin:\s\b'
                                                     r'|\bHair:\s\b|\bEyes:\s\b|\t', x))))

# remove \t
df['users'] = df['users'].apply(lambda x: x.strip())
df['reviews'] = df['reviews'].apply(lambda x: x.strip())
df['dates'] = df['dates'].apply(lambda x: x.strip())

# get lipies score
df['lipies'] = df['lipies'].apply(lambda x: int(x[2]))

# get dates
df['dates'] = df['dates'].apply(lambda x: re.findall('\d+/\d+/\d+', x))
# this will get a list of individual characters
# df['dates_t'] = df['dates'].apply(lambda x: re.findall('\d|/|:', x))
# turn string to datetime object
df['dates'] = df['dates'].apply(lambda x: datetime.strptime(x[0], '%m/%d/%Y'))
# format to month-year
# should keep datetime type and only format when output
# df['dates'] = df['dates'].apply(lambda x: x.strftime('%b-%Y'))

# split char into age, skin, hair and eyes
# https://stackoverflow.com/questions/35491274/pandas-split-column-of-lists-into-multiple-columns
# tolist() is taking a whole column into a list, then turn that list into a df column
df[['age', 'skin', 'hair', 'eyes']] = pd.DataFrame(df.chars.values.tolist(), index= df.index)
# split skin and hair column into a list of string instead of strings
df['skin'] = df['skin'].apply(lambda x: x.split(','))
df['skin'] = df['skin'].apply(lambda x: [i.strip() for i in x])

df['hair'] = df['hair'].apply(lambda x: x.split(','))
df['hair'] = df['hair'].apply(lambda x: [i.strip() for i in x])

# split into more columns
df[['skin_type', 'skin_tone', 'skin_undertone']] = pd.DataFrame(df.skin.values.tolist(), index= df.index)
df[['hair_color', 'hair_type', 'hair_texture']] = pd.DataFrame(df.hair.values.tolist(), index= df.index)


# there is some float in the column (?) so have to convert to str
df['ingredients'] = df['ingredients'].astype(str)


# save df for descriptive stats
dill.dump(df, open('df_clean.pkd', 'wb'))