import pandas as pd


class DataFrameColumnCleaner:
    def __init__(self, strip_whitespace=True, to_numeric_errors="ignore"):
        """
        Initializes the processor with global defaults to:
        - strip_whitespace (bool): Default behavior for removing
        leading/trailing whitespace.
        - to_numeric_errors (str): Default behavior for handling errors in
        numeric conversion. Default is "ignore", but can be set to "coerce".
        """
        self.strip_whitespace = strip_whitespace
        self.to_numeric_errors = to_numeric_errors

    def clean_commas_if_numeric(self, val, strip_whitespace):
        """
        Removes commas from a string if the remaining characters are numeric.
        Leaves non-numeric strings unchanged.
        """
        if isinstance(val, str):
            # Optionally strip whitespace for this column
            if strip_whitespace:
                val = val.strip()

            # Remove commas temporarily to check if it's numeric
            temp_val = val.replace(",", "")

            # Check if remaining characters are numeric (allowing decimals)
            if temp_val.replace(".", "").isdigit():
                return temp_val  # Return cleaned numeric string
            else:
                return val  # Return original value if non-numeric
        return val  # Return original value if not a string

    def process_column(self, col, strip_whitespace, to_numeric_errors):
        """
        Applies the clean_commas_if_numeric function to a column and converts it to numeric.
        Handles conversion errors as per the 'to_numeric_errors' setting for this column.
        """
        col = col.apply(
            lambda x: self.clean_commas_if_numeric(x, strip_whitespace)
        )
        return pd.to_numeric(col, errors=to_numeric_errors)

    def process_dataframe(self, df, column_settings=None):
        """
        Processes each column in the DataFrame with per-column settings.
        If no settings are provided, global defaults are used.

        column_settings (dict):
        Optional dictionary where keys are column names, and values are dicts
        specifying 'strip_whitespace' and 'to_numeric_errors'.
        Example: {"column1": {"strip_whitespace": False,
        "to_numeric_errors": "coerce"}}
        """
        if column_settings is None:
            column_settings = {}

        for col in df.columns:
            # Get per-column settings or use global defaults
            col_settings = column_settings.get(col, {})
            strip_whitespace = col_settings.get(
                "strip_whitespace", self.strip_whitespace
            )
            to_numeric_errors = col_settings.get(
                "to_numeric_errors", self.to_numeric_errors
            )

            # Process the column with the defined settings
            df[col] = self.process_column(
                df[col], strip_whitespace, to_numeric_errors
            )
        return df
