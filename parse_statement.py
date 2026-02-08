import os 
import pandas as pd
import re
import pdfplumber
from datetime import datetime

# Function to extract data from a single PDF
# For extracting TD credit card PDF statements 
def extract_pdf_to_dataframe(pdf_file, statement_type):
    data = []
    # putting the regex pattern outside the read line loop to avoid compiling it every line 
    if statement_type == "TD_Credit": # For TD Credit card pdf statements
    # look for pattern "JAN7 JAN8 PIONEER43417KITCHENER $51.43"
        pattern = re.compile(
            r"""^
            (?P<transaction_date>[A-Z]{3}\s+\d{1,2})
            \s+
            (?P<posting_date>[A-Z]{3}\s+\d{1,2})
            \s+
            (?P<description>.*?)
            \s+
            (?P<amount>-?\$\d{1,3}(?:,\d{3})*\.\d{2})
            """,
            re.VERBOSE
            ) 
    elif statement_type == "BMO_Credit": # For BMO credit card pdf statements 
    # look for pattern "Jul. 29 Jul. 30 RCSS #2822 KITCHENER ON 86.51"
        pattern = re.compile(
            r"""
            (?P<transaction_date>[A-Z][a-z]{2}\.\s\d{1,2})# transaction date
            \s+
            (?P<postting_date>[A-Z][a-z]{2}\.\s\d{1,2})   # post date
            \s+
            (?P<description>.*?)                          # description
            \s+
            (?P<amount>-?\d{1,3}(?:,\d{3})*(?:\.\d{2}))   # amount
            \s*
            (?P<credit>CR)?                               # optional 'CR' flag
            """,
            re.VERBOSE
            ) 
    else:                     
        raise ValueError(f"Invalid statement_type: '{statement_type}'. Must be 'TD' or 'BMO'.")
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1)
            # DEBUG print('pause')
            # Process the text to extract structured data
            # Example: Split lines, find dates, amounts, and descriptions
            for line in text.split("\n"):
                matches = pattern.search(line)
                if matches:
                    # #>>> DEBUG 
                    # print(line)
                    # print(matches)
                    date = matches["transaction_date"]
                    description = matches["description"]
                    amount = float(matches["amount"].replace(",","").replace("$",""))
                    credit_flag = matches.groupdict().get("credit") # deal with TD statement where there is no 'credit'
                    if credit_flag == "CR": amount = -amount # handle the CR flag in BMO statements, representing credit card payment
                    data.append({"date": date, "description": description, "amount": amount, "source": statement_type})
    return pd.DataFrame(data)

# Function to clean up dataframe from pdf file 
#   - convert date to proper datetime format 
#   - set amount to negative for credit transactions 
def clean_up_pdf_data(pdf_df, year): 
    # Cleaning up data from pdf source 
    def convert_MMMDD_to_yyyymmdd(date_str, year):
        date_str = date_str.replace(".", "").replace(" ", "").upper()
        return datetime.strptime(f"{date_str}{year}", "%b%d%Y").strftime("%Y-%m-%d")
    pdf_df['date'] = pdf_df['date'].apply(lambda x: convert_MMMDD_to_yyyymmdd(x, year))
    pdf_df['date'] = pd.to_datetime(pdf_df['date'])
    pdf_df['amount'] = pdf_df['amount'].astype(float)
    pdf_df['amount'] = pdf_df['amount'] * -1
    pdf_df['balance'] = 0.0     # the balanace collumn is used in post-processing for a running sum 

    return pdf_df

# Function to convert multiple pdf into data frame and clean up data: 
# take in pdf statement and convert into dataframe. With following cleanup: 
#   - convert dates to yyyy/mm/dd format
#   - convert amount to float 
#   - convert the amount to negative (i.e. expense)
def convert_multiple_pdf_statements_to_dataframe(pdf_files):    
    # Process multiple PDFs and merge into one DataFrame
    pdf_dataframes = []
    # regex for parsing pdf filename 
    pattern = re.compile(
        r"""
        (?P<statement_type>\w+_\w+)
        _
        (?P<statement_date_start>\d{8})
        _
        (?P<statement_date_end>\d{8})
        (_\w+)?.pdf
        """, 
        re.VERBOSE
    )

    for pdf in pdf_files: 
        filename_matches = pattern.search(os.path.basename(pdf))
        if not filename_matches: raise ValueError(f"Invalid pdf file name: '{pdf}'. Example: TD_Credit_YYYYMMDD_YYYYMMDD.pdf")
        statement_type = filename_matches['statement_type']
        pdf_df = extract_pdf_to_dataframe(pdf, statement_type)
        # clean up df extracted from pdf 
        if filename_matches['statement_date_start'][:4] == filename_matches['statement_date_end'][:4]: 
            statement_year = filename_matches['statement_date_start'][:4]
        else: raise ValueError(f"Invalid pdf file name: '{pdf}'. Statement start and end dates must from the same year. ")
        pdf_df = clean_up_pdf_data(pdf_df, statement_year)

        pdf_dataframes.append(pdf_df) 
    # Combine all DataFrames
    combined_df = pd.concat(pdf_dataframes, ignore_index=True)

    # Calculate balance for each credit card transaction
    starting_balances = {
    "BMO_Credit": -1438.39,
    "TD_Credit": 0.00
    }
    # Sort by date to ensure balance is calculated chronologically
    combined_df = combined_df.sort_values(by="date").reset_index(drop=True)
    # Apply running balance by 'source'
    combined_df["balance"] = combined_df.groupby("source")["amount"].transform(
        lambda x: x.cumsum() + starting_balances.get(x.name, 0)
    )

    return combined_df

# Function to read and concat multiple csv statements from TD
# The csv must:
#   - no header row 
#   - columns: date, description, debit, credit, balance
#   - date format upported: yyyy-mm-dd, mm/dd/yyyy 
def cleanup_multiple_csvs(csv_files): 
    combined_csv_df = []
    for csv in csv_files: 
        csv_df = pd.read_csv(csv, header=None)
        csv_df.columns = ['date','description','withdraw','deposit','balance']
        csv_df = csv_df.fillna(0)
        csv_df['date'] = pd.to_datetime(csv_df['date'], errors="coerce", format="mixed", dayfirst=False)
        csv_df['amount'] = csv_df['deposit'] - csv_df['withdraw'] 
        statement_type = os.path.basename(csv).split("_")
        if statement_type[1] == "Credit": # balance on credit card statements should be negative 
            csv_df['balance'] = csv_df['balance'] * -1 
        csv_df['source'] = f"{statement_type[0]}_{statement_type[1]}" # e.g. "TD_Checking" or "TD_Credit"
        combined_csv_df.append(csv_df[['date', 'description', 'amount', 'source','balance']])
    return pd.concat(combined_csv_df, ignore_index=True)

def main(): 
    # specify folder that contains all transactions pdf or csv 
    # input data must follow naming convention to allow script to ingest metadata 
    # <Bank Name>_<Account type>_<Start date, yyyymmdd>_<End date, yyyymmdd>.csv/pdf
    # e.g. BMO_Credit_20251025_20251124 
    # SPECIAL for PDF: a single file cannot contain transactions from 
    folder_path = 'raw'
    # Convert and merge PDF files 
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
    if len(pdf_files) > 0: 
        pdf_out_df = convert_multiple_pdf_statements_to_dataframe(pdf_files)

    # Merge csv files 
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]
    if len(csv_files) > 0: 
        csv_out_df = cleanup_multiple_csvs(csv_files)

    # Save to a single CSV file
    out_df = pd.concat([csv_out_df,pdf_out_df], ignore_index=True)
    out_df = out_df.drop_duplicates()
    out_df = out_df.sort_values(by="date").reset_index(drop=True)
    out_df.to_csv("merged_statements.csv", index=False)
    print("Merged CSV created successfully!")

if __name__ == "__main__":
    main()