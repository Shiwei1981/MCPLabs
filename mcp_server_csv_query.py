import argparse
from typing import List, Dict, Any, Optional
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from glob import glob
import polars as pl
import pandas as pd  
import sqlite3  


parser = argparse.ArgumentParser()
parser.add_argument("--file_location", type=str, default="data/*.csv")
args = parser.parse_args()


mcp = FastMCP("MCPServerCSVQuery", dependencies=["polars"])


@mcp.tool()
def get_files_list() -> str:
    """
    Get the list of files that are source of data
    """
    files_list = glob(args.file_location)
    return files_list


def read_file(
    file_location: str,
    file_type: str = Field(
        description="The type of the file to be read. Supported types are csv and parquet",
        default="csv",
    ),
) -> pl.DataFrame:
    """
    Read the data from the given file location
    """
    if file_type == "csv":
        return pl.read_csv(file_location)
    elif file_type == "parquet":
        return pl.read_parquet(file_location)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def read_file_list(
    file_locations: List[str],
    file_type: str = Field(
        description="The type of the file to be read. Supported types are csv and parquet",
        default="csv",
    ),
) -> pl.DataFrame:
    """
    Read the data from the given file locations
    """
    if file_type == "csv":
        dfs = []
        for file_location in file_locations:
            dfs.append(pl.read_csv(file_location))
        return pl.concat(dfs)
    elif file_type == "parquet":
        dfs = pl.read_parquet(file_locations)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


@mcp.tool()
def get_schema(
    file_location: str,
    file_type: str = Field(
        description="The type of the file to be read. Supported types are csv and parquet",
        default="csv",
    ),
) -> List[Dict[str, Any]]:
    """
    Get the schema of a single data file from the given file location
    """
    df = read_file(file_location, file_type)

    schema = df.schema
    schema_dict = {}
    for key, value in schema.items():
        schema_dict[key] = value
    return schema_dict


polars_sql_aggregate_functions = [
    "avg",
    "count",
    "first",
    "last",
    "max",
    "median",
    "min",
    "sum",
    "quantile_count",
    "quantile_disc",
    "stddev",
    "sum",
    "variance",
]

polars_sql_array_functions = [
    "array_agg",
    "array_contains",
    "array_get",
    "array_length",
    "array_lower",
    "array_mean",
    "array_reverse",
    "array_sum",
    "array_to_string",
    "array_unique",
    "array_upper",
    "unnest",
]

polars_sql_bitwise_functions = [
    "bit_and",
    "bit_count",
    "bit_or",
    "bit_xor",
]

polars_sql_conditional_functions = [
    "coalesce",
    "greatest",
    "if",
    "ifnull",
    "least",
    "nullif",
]

polars_sql_mathematical_functions = [
    "abs",
    "cbrt",
    "ceil",
    "div",
    "exp",
    "floor",
    "ln",
    "log2",
    "log10",
    "mod",
    "pi",
    "pow",
    "round",
    "sign",
    "sqrt",
]

polars_sql_string_functions = [
    "bit_length",
    "concat",
    "concat_ws",
    "date",
    "ends_with",
    "initcap",
    "left",
    "length",
    "lower",
    "ltrim",
    "normalize",
    "octet_length",
    "regexp_like",
    "replace",
    "reverse",
    "right",
    "rtrim",
    "starts_with",
    "strpos",
    "strptime",
    "substr",
    "timestamp",
    "upper",
]

polars_sql_temporal_functions = [
    "date_part",
    "extract",
    "strftime",
]

polars_sql_type_functions = [
    "cast",
    "try_cast",
]

polars_sql_trigonometric_functions = [
    "acos",
    "acosd",
    "asin",
    "asind",
    "atan",
    "atand",
    "atan2",
    "atan2d",
    "cot",
    "cotd",
    "cos",
    "cosd",
    "degrees",
    "radians",
    "sin",
    "sind",
    "tan",
    "tand",
]

polars_sql_functions = {
    "aggregate": polars_sql_aggregate_functions,
    "array": polars_sql_array_functions,
    "bitwise": polars_sql_bitwise_functions,
    "conditional": polars_sql_conditional_functions,
    "mathematical": polars_sql_mathematical_functions,
    "string": polars_sql_string_functions,
    "temporal": polars_sql_temporal_functions,
    "type": polars_sql_type_functions,
    "trigonometric": polars_sql_trigonometric_functions,
}


def gen_polars_sql_functions_str():
    sql_functions_agg = []
    for sql_fn_category, sql_fns in polars_sql_functions.items():
        sql_fns_str = ["- " + sql_fn.capitalize() for sql_fn in sql_fns]
        sql_fns_str = "\n".join(sql_fns_str)
        sql_fn_with_header = f"{sql_fn_category.capitalize()}: \n{sql_fns_str}\n\n"
        sql_functions_agg.append(sql_fn_with_header)
    return "\n".join(sql_functions_agg)


query_description = f"""
Supports the use of standard SQL syntax. In the query, use the filename of the CSV as the table name replace "self" in query script, and making sure to exclude the file extension for clarity.
Supported functions are:
{gen_polars_sql_functions_str()}
"""


@mcp.tool()
def execute_polars_sql(
    file_locations: List[str],
    query: str = Field(
        description=query_description,
    ),
    file_type: str = Field(
        description="The type of the file to be read. Supported types are csv and parquet",
        default="csv",
    ),
) -> List[Dict[str, Any]]:
    """
    Reads the data from the given file locations. Note that file_locations
    can be a list of multiple files. However, all files must have the same schema
    and the same columns. Executes the given polars sql query and returns the result.
    Note that the polars sql query must use the table name as `self` to refer to the source data.
    
    df = read_file_list(file_locations, file_type)
    op_df = df.sql(query)
    output_records = op_df.to_dicts()
    return output_records
    """ 


    # 创建一个 SQLite 数据库 in-memory  
    conn = sqlite3.connect(':memory:')  
    
    # 遍历每个 CSV 文件并加载到 SQLite 数据库中  
    for filename in file_locations:  
        df = pd.read_csv(filename)  
        
        # 将 DataFrame 存入 SQLite 数据库，表名使用文件名（去掉扩展名）  
        table_name = filename.split('\\')[-1].replace('.csv', '')  
        df.to_sql(table_name, conn, index=False, if_exists='replace')  
    
    # 执行 SQL 查询  
    result = pd.read_sql_query(query, conn)  
    conn.close()  
    # 输出查询结果  
    print(result)  
    return result
    # 关闭数据库连接 


def main():
    mcp.run()


if __name__ == "__main__":
    main()
