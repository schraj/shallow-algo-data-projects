import pandas as pd

class Eda:
  file_path = ''
  df = None

  def __init__(self, file_path):
    self.file_path = file_path
  
  def print_n_lines(self, count):
    with open(self.file_path, 'r') as lines:
      for _ in range(count):
        print(next(lines))

  def set_df(self):
    self.df = pd.read_csv(self.file_path)

  def value_counts(self, exclude_columns = []):
    columns = [c for c in self.df.select_dtypes(include=['object', 'bool']) if c not in exclude_columns]
    columns.extend([c for c in self.df.select_dtypes(include=['int64']) if c not in exclude_columns])
    for column in columns:
        print(f"Value counts for '{column}':")
        print(self.df[column].value_counts())
        print()
  
  def transform_dates(self, date_text_fields = []):
    for field in date_text_fields:
      self.df['DT_' + field] = pd.to_datetime(self.df[field])

  def pivot_tables(self, target_column, id_column, exclude_columns = []):
    columns = [c for c in self.df.columns if c not in exclude_columns]
    for column in columns:
        print(f"Pivot for '{column}' against '{target_column}'")
        pivoted_df = self.df.pivot_table(index=target_column, columns=column, values=id_column, aggfunc='count')
        print(pivoted_df)
        print()

  def fill_nominal_missing(self, columns = []):
    for column in columns:
      self.df[column] = self.df[column].fillna('Missing')

