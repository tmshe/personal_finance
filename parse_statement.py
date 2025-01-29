import os 
import pandas as pd
import re
import pdfplumber
from datetime import datetime

# Function to extract data from a single PDF
def extract_pdf_to_dataframe(pdf_file):
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1)
            # DEBUG print('pause')
            # Process the text to extract structured data
            # Example: Split lines, find dates, amounts, and descriptions
            for line in text.split("\n"):
                # look for pattern "JAN7 JAN8 PIONEER43417KITCHENER $51.43"
                # Suitable for TD Credit card pdf statements
                pattern = r"^(\w{3} \d+)\s+(\w{3} \d+)?\s+([\w\s\S]+?)\s+(-?\$[\d,]+\.\d{2})" 
                matches = re.findall(pattern, line)
                if matches:
                    # #>>> DEBUG 
                    # print(line)
                    # print(matches)
                    date = matches[0][0]
                    description = matches[0][2]
                    amount = matches[0][3].replace(",","").replace("$","") 
                    data.append({"date": date, "description": description, "amount": amount})
    return pd.DataFrame(data)

# Function to convert multiple pdf into data frame and clean up data: 
# take in pdf statement and convert into dataframe. With following cleanup: 
#   - convert dates to yyyy/mm/dd format
#   - convert amount to float 
#   - convert the amount to negative (i.e. expense)
# !!CAUTION!!: the script handles statements from a single year. Manual input is needed if pdf statemt contains transactions from two years (DEC and JAN)
def convert_multiple_pdf_statements_to_dataframe(pdf_files,year): 
    # Process multiple PDFs and merge into one DataFrame
    all_dataframes = [extract_pdf_to_dataframe(pdf) for pdf in pdf_files]

    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Cleaning up data from pdf source 
    def convert_MMMDD_to_yyyymmdd(date_str, year):
        return datetime.strptime(f"{date_str}{year}", "%b %d%Y").strftime("%Y/%m/%d")
    combined_df['date'] =  combined_df['date'].apply(lambda x: convert_MMMDD_to_yyyymmdd(x, year))
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['amount'] = combined_df['amount'].astype(float)
    combined_df['amount'] = combined_df['amount'] * -1

    return combined_df

def cleanup_multiple_csvs(csv_files): 
    all_dataframes = [pd.read_csv(csv, header=None) for csv in csv_files]
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.columns = ['date','description','withdraw','deposit','balance']
    combined_df = combined_df.fillna(0)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['amount'] = combined_df['deposit'] - combined_df['withdraw'] 

    return combined_df[['date','description','amount']]

def main(): 
    # specify folder that contains all transactions pdf or csv 
    folder_path = 'raw'
    # Convert and merge PDF files 
    # PDF statements does not have the year, specify the year the statements are for. 
    # DO NOT USE pdf statemts containing transactions from two years (e.g. 2023 DEC and 2024 JAN)
    year = 2024
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
    pdf_out_df = convert_multiple_pdf_statements_to_dataframe(pdf_files,year)
    
    # Merge csv files 
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
    csv_out_df = cleanup_multiple_csvs(csv_files)

    # Save to a single CSV file
    out_df = pd.concat([csv_out_df,pdf_out_df], ignore_index=True)
    out_df.to_csv("merged_statements.csv", index=False)
    print("Merged CSV created successfully!")

if __name__ == "__main__":
    main()