import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from webdriver_manager.chrome import ChromeDriverManager
import csv
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_csv(file_path):
    # Create the csv file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Create the "title" row
        writer.writerow(['Brand', 'WaterProof', 'Gender', 'TimeDisplay', 'Chronograph', 'Rating', 'Price'])


def next_page(driver):
    # Function that swipe between pages
    next_button = driver.find_element(By.CSS_SELECTOR, '.page-item.next')
    next_button.click()


def get_elements_value(driver, file):
    try:
        # Collecting the parameter values from the HTML
        brand = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent > '
                                                     'div.model-page-bg > div > div.specificationContainer > '
                                                     'div.right > div:nth-child(2) > div > div.ParamColValue')
        waterproof = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent > '
                                                          'div.model-page-bg > div > div.specificationContainer > '
                                                          'div.left > div:nth-child(2) > div > div.ParamColValue')
        gender = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent >'
                                                      'div.model-page-bg > div > div.specificationContainer > '
                                                      'div.right > div:nth-child(3) > div > div.ParamColValue')
        timedisplay = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent'
                                                           '> div.model-page-bg > div > div.specificationContainer > '
                                                           'div.left > div:nth-child(1) > div > div.ParamColValue')
        chronograph = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent >'
                                                           'div.model-page-bg > div > div.specificationContainer > '
                                                           'div.left > div:nth-child(3) > div > div.ParamColValue')
        price = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent > '
                                                     'div.model-details-bg.FullSceenContent > div > '
                                                     'div.model-details-wrapper > div.model-page-details >'
                                                     'div.prices-txt > span:nth-child(1)')
        rating = driver.find_element(By.CSS_SELECTOR, 'body > div.MainDiv > div.main-bg > div.MainContent > div.model-details-bg.FullSceenContent > div > div.model-details-wrapper > div.model-page-details > div.scores-container > a.rating-line > div.rate-cnt')

        # Writes the values to the csv file
        writer = csv.writer(file)
        writer.writerow([brand.text, waterproof.text, gender.text, timedisplay.text, chronograph.text, rating.text,  price.text])

    except Exception:
        pass


def more_info(driver, file_path):
    for p in range(400):
        print(p)
        # Create list of all element in the HTML of the page
        brand_elements = driver.find_elements(By.CSS_SELECTOR, 'div.withModelRow.ModelRowContainer')
        with open(file_path, 'a', newline='') as file:

            # For each element click on the more-information button
            for i in range(len(brand_elements)):
                brand_elements = driver.find_elements(By.CSS_SELECTOR, 'div.withModelRow.ModelRowContainer')
                brand = brand_elements[i]

                try:
                    more_info_button = brand.find_element(By.CSS_SELECTOR, 'a.more-details')
                    delay = random.randint(1, 3)
                    time.sleep(delay)
                    more_info_button.click()
                    get_elements_value(driver, file)
                except Exception:
                    pass

                # After clicking "more-info" runs the get_elements_value function
                driver.back()
        # Runs the next_page function to go to the next page
        next_page(driver)
        delay = random.randint(2, 4)
        time.sleep(delay)


def collecting_data():
    # Function that unite all function and collect the data
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    website_url = 'https://www.zap.co.il/models.aspx?sog=g-watch&pageinfo'
    file_path = 'C:\dataProject\CollectDataWatches.csv'
    driver.get(website_url)
    create_csv(file_path)
    more_info(driver, file_path)


def cleaning_data():
    file_path = 'C:\dataProject\CollectDataWatches.csv'

    # Reading from csv
    data = pd.read_csv(file_path, encoding='cp1255')

    # Adjust information to research
    data['Price'] = data['Price'].str.replace('₪', '')
    data['Price'] = data['Price'].str.replace('[^\d.]', '').str.replace(',', '').astype(float)

    # Convert non-numeric values to NaN
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    # Convert the 'Price' column to float
    data['Price'] = data['Price'].astype(float)

    # Change information for EDA
    data['WaterProof'] = data['WaterProof'].str.replace('לא זמין', '0')
    data['WaterProof'] = data['WaterProof'].str.replace('עמיד במים', '50')
    data['Rating'] = data['Rating'].str.replace('חוות דעת אחת', '1')
    data['TimeDisplay'] = data['TimeDisplay'].str.replace('משולב, דיגיטלי', 'משולב')
    data['Chronograph'] = data['Chronograph'].replace({'לא כולל': 0, 'כולל': 1, 'כולל, לא כולל': 2})

    # Change hebrew to english
    data['Gender'] = data['Gender'].str.replace('לאישה', 'women')
    data['Gender'] = data['Gender'].str.replace('לגבר', 'man')
    data['Gender'] = data['Gender'].str.replace('לילדים', 'kids')
    data['TimeDisplay'] = data['TimeDisplay'].str.replace('אנלוגי', 'analog')
    data['TimeDisplay'] = data['TimeDisplay'].str.replace('דיגיטלי', 'digital')
    data['TimeDisplay'] = data['TimeDisplay'].str.replace('משולב', 'combine')

    # Deleting Words In Numeric
    data['WaterProof'] = data['WaterProof'].str.extract('(\d+)')
    data['WaterProof'] = pd.to_numeric(data['WaterProof'])
    data['Rating'] = data['Rating'].str.extract('(\d+)')
    data['Rating'] = pd.to_numeric(data['Rating'])

    # Deleting irrelevant raw
    data = data[~data.apply(lambda row: row.astype(str).str.contains('לא רלוונטי').any(), axis=1)]
    data = data[~data.apply(lambda row: row.astype(str).str.contains('לא זמין').any(), axis=1)]
    data = data[~data.apply(lambda row: row.astype(str).str.contains('יעודכן בקרוב').any(), axis=1)]

    # Clean duplicate
    data = data.drop_duplicates()

    # Drop rows with missing or non-numeric values in the 'Price' column
    data = data.dropna()

    # Import the "clean data" to a new file
    cleaned_file_path = 'C:\dataProject\CleanedData.csv'
    data.to_csv(cleaned_file_path, encoding='cp1255', index=False)


def data_visualization():

    data = pd.read_csv('C:\dataProject\CleanedData.csv', encoding='cp1255')
    plt.style.use('dark_background')

    categories = ['Brand', 'WaterProof', 'Gender', 'TimeDisplay', 'Chronograph']
    for category in categories:

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=category, y='Price', data=data)
        ax.set_title(f'Average Price by {category}', fontsize=16)
        ax.set_xlabel(category, fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.tick_params(axis='x', labelrotation=45,
                       labelsize=7)
        ax.tick_params(axis='y', labelsize=10)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.xticks(rotation=45)
        plt.title(f'Price Distribution by {category}')
        sns.boxplot(x=category, y='Price', data=data)
        plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='WaterProof', y='Price', data=data)
    plt.title('Water Resistance Depth vs. Price')
    plt.xlabel('Water Resistance Depth')
    plt.ylabel('Price')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Gender', y='Price', data=data)
    plt.title('Gender vs. Price')
    plt.xlabel('Gender')
    plt.ylabel('Price')
    plt.show()

    # Plotting histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Price'], kde=False)
    plt.title('Price Distribution (Histogram)')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.show()

    # Changes data to numeric for heat map
    data['Gender'] = data['Gender'].replace(
        {'women': 0, 'man': 1, 'kids': 2, 'women, man': 3, 'kids, women': 4, 'kids, man': 4})
    data['TimeDisplay'] = data['TimeDisplay'].replace(
        {'digital': 0, 'analog': 1, 'combine': 2})
    cleaned_file_path = 'C:\dataProject\CleanedData.csv'
    data.to_csv(cleaned_file_path, encoding='cp1255', index=False)

    # Create the heatmap
    cols = ['Price', 'Rating', 'Chronograph', 'WaterProof', 'Gender', 'TimeDisplay']
    heatmap_data = data[cols]
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data.corr(), annot=True)
    plt.title('Correlation Heatmap')
    plt.show()


def machine_learning():

    data = pd.read_csv('C:\dataProject\CleanedData.csv', encoding='cp1255')


    X = data[['WaterProof', 'Gender', 'TimeDisplay', 'Chronograph', 'Rating']]
    y = data['Price']

    # Linear Regression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Linear Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Linear Regression - Actual vs. Predicted Prices")
    plt.show()

    # Decision Tree

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Decision Tree Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Decision Tree Regression - Actual vs. Predicted Prices")
    plt.show()

    # Random forest

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Random Forest Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Random Forest Regression - Actual vs. Predicted Prices")
    plt.show()

    # KNeighbors

    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("KNeighborsRegressor:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("KNeighborsRegressor - Actual vs. Predicted Prices")
    plt.show()


def project_code():

    collecting_data()
    cleaning_data()
    data_visualization()
    machine_learning()


def main():
    project_code()


if __name__ == '__main__':
  main()
