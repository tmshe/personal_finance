import pandas as pd
df_raw = pd.read_csv("2025_statements_classfied.csv")
df_mapping = pd.read_csv("category_mapping.csv")
## Clean 
# Parse date
df_raw["date"] = pd.to_datetime(df_raw["date"])
# Convert amount from string like "-$50.79" to float with 2 decimal places
df_raw["amount"] = (
    df_raw["amount"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
).round(2)

# Ensure month is treated consistently (YYYYMM as string)
df_raw["month"] = df_raw["month"].astype(str)

## Create monthly cashflow table 
def cash_flow_pivot(df: pd.DataFrame, 
    index="category", 
    columns="month",
    values="amount",
    add_total=True, 
    use_category_group=False, 
    category_mapping=None
    ): 

    working_df = df.copy()

    # Apply category grouping if requested
    if use_category_group:
        if category_mapping is None:
            raise ValueError("category_mapping must be provided when use_category_group=True")
        working_df["category_group"] = working_df["category"].map(category_mapping)
        working_df["category_group"] = working_df["category_group"].fillna("Other")
        index = "category_group"

    # Build pivot
    pivot = working_df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc="sum",
        fill_value=0
    )

    # Sort columns chronologically/alphabetically
    pivot = pivot.reindex(sorted(working_df[columns].astype(str).unique()), axis=1)

    # Add total column
    if add_total:
        pivot["Total"] = pivot.sum(axis=1)
        pivot.loc["Monthly Total"] = pivot.sum(axis=0)

    return pivot

## Separate household, personal, and rental 
# Rental = category starts with "Rental "
is_rental = df_raw["category"].str.startswith("Rental", na=False)
df_rental = df_raw[is_rental].copy()

# Household transactions dataframe = BMO credit and Joint checking 
household_sources = ["BMO_Credit", "TD_JointChecking"]
df_household = df_raw[df_raw["source"].isin(household_sources)].copy()

# Personal transactions (everything else)
df_personal = df_raw[~df_raw["category"].str.startswith("Rental") & ~df_raw["source"].isin(household_sources)].copy()

# Load and apply category mapping 
CATEGORY_GROUP_MAP = dict(zip(df_mapping["category"], df_mapping["category_group"]))
df_household["category_group"] = df_household["category"].map(CATEGORY_GROUP_MAP)
df_personal["category_group"] = df_personal["category"].map(CATEGORY_GROUP_MAP)

rental_cashflow = cash_flow_pivot(df_rental)
# print("Rental Cashflow")
# print(rental_cashflow)

household_cashflow = cash_flow_pivot(
    df_household, 
    use_category_group=True, 
    category_mapping=CATEGORY_GROUP_MAP
    )

personal_cashflow = cash_flow_pivot(
    df_personal, 
    use_category_group=True, 
    category_mapping=CATEGORY_GROUP_MAP
    )
# print("Personal Cashflow")
# print(personal_cashflow)



print('end')