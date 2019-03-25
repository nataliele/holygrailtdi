import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4
from bs4 import BeautifulSoup
# from selenium import webdriver
import time
import re
import math
import string
import collections
import dill


# get working directory to save files
working_dir = os.getcwd()


# helper variables
base_url = 'https://www.makeupalley.com'
# have to use https because using http will make it default to the first review page
part_prod_link = '/product/showreview.asp/ItemId='
part_brand_link = '/product/searching.asp/Brand='


# Category names and ids under skincare-face; make list so dont have to parse them for every page
cat_name = ['Moisturizers', 'Masks', 'Scrubs', 'Treatments (Eye)', 'Neck/Decollete Cream', 'Eye Makeup Remover', 'Treatments (Face)', 'Toners', 'Cleansers']
cat_id = ['707', '703', '704', '706', '709', '708', '705', '702', '701']
cat_dict = dict(zip(cat_id, cat_name))



def parse_search_page(url):
    """
    function to go to each search page and parse data
    :param url: a link of a search page eg. https://www.makeupalley.com/product/searching.asp/CategoryId=708/NumberOfReviews=5/SC=HOWMANY/SD=DESC/page=2'
    :return: list of all product links on the search page, their product id and the number of review for each product
    """
    soup = go_link(url)

    ### get list of product links on the page
    modules = soup.find_all('a')

    # extract links from web elements
    module_list = [i.get('href') for i in modules]
    # had 'argument of type 'NoneType' is not iterable' error because there are items that are not links in the list
    # solution is to wrap item in str()
    module_list1 = [i for i in module_list if (part_prod_link in str(i))]

    # links are duplicated so get a unique list
    # module_list2 = list(set(module_list1))
    # set doesnt reserve order and is random so cant use
    # should find another way that's less prone to error than slicing
    module_list2 = module_list1[::2]

    module_list3 = [make_prod_link(i) for i in module_list2]

    ### get id of product for identification
    # product_ids = [make_prod_link(i)[1] for i in module_list2]

    ### get list of number of reviews for each of the product on the page
    parts = soup.find_all('td', class_='mobile-control')
    # this works because class is an attribute of the 'td' tag
    nums = [i.text for i in parts]
    num_review = nums[1::2]
    # each page displays 10 reviews
    # some strings are in the form '1,399' so have to remove comma and keep only digits
    num_review_digits = [re.sub("\D", "", s) for s in num_review]
    num_page = [math.ceil(int(i) / 10) for i in num_review_digits]

    # one (?) page have problems beyond 66 but cant fix it here because other pages are fine
    # num_page = [min(i, 66) for i in num_page]

    return module_list3, num_page


def go_link(link):
    """
    function to get soup version of a html link
    :param link: a html link
    :return: beautifulsoup object
    """
    r = requests.get(link)
    html_doc = r.text
    return BeautifulSoup(html_doc, "html5lib")



# need to cut part of product link, add page= to be able to traverse product pages with more than 10 reviews
def make_prod_link(link):
    start = part_prod_link
    end = '/'
    first_part = link.find(start)+len(start)
    link_second = link[first_part:]
    second_part = link_second.find(end)
    new_link = link[:(first_part+second_part)] + '/page='
    # id = new_link[first_part:(first_part + second_part)]
    return new_link




# function to find brand id
def find_id(link, part_link):
    """
    find id in a href
    :param link: eg. '/product/showreview.asp/ItemId=66238/page=1'
    :param part_link: '/product/showreview.asp/ItemId='
    :return: 66238
    """
    start = part_link
    end = '/'
    first_part = link.find(start) + len(start)
    link_second = link[first_part:]
    second_part = link_second.find(end)
    new_link = link[:(first_part + second_part)]
    id = new_link[first_part:(first_part + second_part)]
    return id


def go_prod(link):
    """
    Function go to each product link and get data
    :param link: string that represents the latter part of the link to product review pages
    :return: variables related to product reviews
    """

    soup = go_link(base_url+link)

    if len(soup.select('title')) == 0:
        pass
    else:
        # find users, characteristics and reviews; and ratings and dates
        users = soup.find_all(class_='user-name')
        user_list = [i.text for i in users]
        # dont have to go to each user and see how many reviews they have because we basically dont have a minimum for users

        chars = soup.find_all(class_='important')
        char_list = [i.text for i in chars]

        reviews = soup.find_all(class_='comment-content')
        review_list = [i.text for i in reviews]

        # added 1/7/2019
        # find user rating/lipies for product
        # lipies = soup.find_all('div', attrs = {'class': 'lipies'})
        # this doesnt work because there are some pages with 2 span tags
        lipies = soup.select('div.lipies span[class*="l-"]')
        # span is a tag, class is an attribute, choose only attribute with value starts with l-
        lipie_list = [i['class'][0] for i in lipies]
        # lipie_list = [child['class'][0] for i in lipies for child in i.children if type(child) is not bs4.element.NavigableString]

        # find review dates
        # dates = soup.find_all(class_= 'time')
        # this doesnt work because some class is called date
        # this is 'or'
        # date = soup.select('div.date, time')
        date = soup.select('div.date')
        # match either div.date or time tag
        date_list = [i.text for i in date]
        # added 1/7/2019

        # number of reviews on each page
        n = len(user_list)

        # ### get id of product for identification
        prod_id = find_id(link, part_prod_link)
        prod_id = [prod_id] * n

        # find name of product
        name_list = soup.find_all('h1')
        name = name_list[1].text
        name = [name] * n

        # find rating of product
        rating = soup.find('h3').text
        rating = [rating] * n

        # find_all() with no parameter will give us the whole document
        # each line will be fed into the function to be evaluated

        repurchase = soup.find_all('p', string=re.compile('would repurchase'))[0].text
        repurchase = [repurchase] * n

        pkg_qual = soup.find_all('p', string=re.compile('Package Quality'))[0].text
        pkg_qual = [pkg_qual] * n

        price = soup.find_all('p', string=re.compile('Price'))[0].text
        price = [price] * n

        ingredient = soup.find_all(id='hold-ingredients')[0].text
        ingredient = [ingredient] * n

        # find brand of product
        brand_list = soup.find_all(class_ = "track_BreadCrumbs_Brand")
        brand = brand_list[0].text
        brand = [brand] * n

        brand_href = brand_list[0].get('href')
        brand_id = find_id(brand_href, part_brand_link)
        brand_id = [brand_id] * n

        return name, rating, repurchase, pkg_qual, price, ingredient, user_list, char_list, review_list, lipie_list, date_list, brand, brand_id, prod_id


# a, b, c, d, e, f, g, h, k, l, m, n, o, p = go_prod('/product/showreview.asp/ItemID=12293/page=2/')



def go_search_page(search_page):
    """
    function to go to search page and get all review data
    :param search_page: url of a search page, for eg. 'https://www.makeupalley.com/product/searching.asp/CategoryId=708/NumberOfReviews=5/SC=HOWMANY/SD=DESC/page=2'
    :return: dataframe with all data for the search page; this dataframe is to be appended with other dataframes from other sesarch pages
    """
    product_list, n_page = parse_search_page(search_page)

    print(len(product_list), len(n_page))

    # get all reviews for first page of search
    # loop through product links

    prod_ids = []
    names = []
    ratings = []
    repurchases = []
    pkg_quals = []
    prices = []
    ingredients = []
    users = []
    chars = []
    reviews = []
    lipies = []
    dates = []
    brands = []
    brand_ids = []

    # test = product_list[:2]

    for i, link in enumerate(product_list):
        # when testing, cant have enumerate because it will parse the string of module_list3[0]
        # for i, link in enumerate(test):
        # loop through review pages
        for j in range(n_page[i]):
            try:
                # id = pd_ids[i]
                # print(product_ids[i])
                # print(link+str(j+1))
                # range starts at 0 so add 1 to start from 1
                # confirm that the code above prints all the review pages for all the products in 1 search page
                full_link = link + str(j + 1)
                if full_link == '/product/showreview.asp/ItemId=4375/page=67':
                    pass
                else:
                    a, b, c, d, e, f, g, h, k, l, m, n, o, p = go_prod(full_link)

                    # needs to be in this loop so append for each review page and not each product
                    prod_ids.append(p)
                    names.append(a)
                    ratings.append(b)
                    repurchases.append(c)
                    pkg_quals.append(d)
                    prices.append(e)
                    ingredients.append(f)
                    users.append(g)
                    chars.append(h)
                    reviews.append(k)
                    lipies.append(l)
                    dates.append(m)
                    brands.append(n)
                    brand_ids.append(o)
            except:
                print('error for' + link + j)
                pass

    # var_list = [prod_ids, names, ratings, repurchases, pkg_quals, prices, ingredients, brands, brand_ids, users, chars,
    #             reviews]
    # sub_var_list = [users, chars, reviews]

    # flatten list of lists of reviews for dataframe
    try:
        prod_ids_flat = [item for sublist in prod_ids for item in sublist]
        names_flat = [item for sublist in names for item in sublist]
        ratings_flat = [item for sublist in ratings for item in sublist]
        repurchases_flat = [item for sublist in repurchases for item in sublist]
        pkg_quals_flat = [item for sublist in pkg_quals for item in sublist]
        prices_flat = [item for sublist in prices for item in sublist]
        ingredients_flat = [item for sublist in ingredients for item in sublist]
        brands_flat = [item for sublist in brands for item in sublist]
        brand_ids_flat = [item for sublist in brand_ids for item in sublist]
        users_flat = [item for sublist in users for item in sublist]
        chars_flat = [item for sublist in chars for item in sublist]
        reviews_flat = [item for sublist in reviews for item in sublist]
        lipies_flat = [item for sublist in lipies for item in sublist]
        dates_flat = [item for sublist in dates for item in sublist]

    # recode empty values, why doesn this work?
    # for i in range(len(var_list)):
    #     var_list[i] = [np.NaN if j=='' else j for j in var_list[i]]

        ingredients_flat = [np.NaN if i == '' else i for i in ingredients_flat]

        df = pd.DataFrame(
            {'pro_ids': prod_ids_flat, 'names': names_flat, 'ratings': ratings_flat, 'repurchases': repurchases_flat
                , 'pkg_quals': pkg_quals_flat, 'prices': prices_flat, 'ingredients': ingredients_flat
                , 'brands': brands_flat, 'brand_ids': brand_ids_flat
                , 'users': users_flat, 'chars': chars_flat, 'reviews': reviews_flat
                , 'lipies': lipies_flat, 'dates': dates_flat})
    except:
        print('error for' + search_page)
        print(len(prod_ids_flat), len(lipies_flat), len(dates_flat))
        print(lipies_flat)
        print(dates_flat)
        pass

    return df



def go_cat_page(url):
    """
    Function to go to each category search page and parse data.
    First, get number of search pages. 
    :param url: a link of a search page eg. 'https://www.makeupalley.com/product/searching.asp/CategoryId=708/NumberOfReviews=5/SD=DESC/SC=HOWMANY/page=2'
    :return: list of all product links on the search page, their product id and the number of review for each product
    """

    df = pd.DataFrame()
    # go to the main category search page
    real_url = url + '1'
    soup = go_link(real_url)
    try:
        # get total number of reviews in the category
        page_links = soup.find_all('a', class_ = 'track_Paging_')
        # put in try except for when there's only 1 search page
        # also slicing is not error-proof
        last_page = page_links[-2].get('href')
        last_page_num = find_id(base_url + last_page, url)
        num_page_review = min(int(last_page_num), 66)

        # because we start from page=1, the last item of range(17) is 16, 16+1 = 17
        for i in range(int(num_page_review)):
            df_page = go_search_page(url + str(i+1))

            # append to main dataframe
            df = df.append(df_page)

    except:
        df_page = go_search_page(url + '1')
        df = df.append(df_page)

    return df

#########################################################################


# add search pages together; need to account for index for category_id

base_search_url = 'https://www.makeupalley.com/product/searching.asp/CategoryId={}/NumberOfReviews=5/SD=DESC/SC=HOWMANY/page='

df = pd.DataFrame()
for i in range(len(cat_id)):
# for i in range(4):
    #subtitute cat_id into base_search_url
    cat_link = base_search_url.format(cat_id[i])
    print(cat_link)

    df_cat = go_cat_page(cat_link)
    print(len(df_cat))
    # add category id
    df_cat['cat_id'] = cat_id[i]

    # save the review data of the cat
    dill.dump(df_cat, open('df_{}.pkd'.format(cat_id[i]), 'wb'))

    df = df.append(df_cat)