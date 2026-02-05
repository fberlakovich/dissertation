from pathlib import Path
import json
import logging
import sys

import polars as pl


def convert_to_int_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert all columns in the input Polars DataFrame that are fully convertible to
    integers. Examples: "1" -> 1, 5.0 -> 5. If any non-null value in a column is not
    an integer when parsed as a float (e.g., "5.2"), the column is left unchanged.

    The function preserves the original column order and names.
    """
    if df.is_empty() or len(df.columns) == 0:
        return df

    converted = {}
    for col in df.columns:
        s = df[col]
        try:
            # Try permissive float casting depending on dtype
            if s.dtype == pl.Utf8:
                # Trim whitespace before casting
                s_float = s.str.strip_chars().cast(pl.Float64, strict=False)
            else:
                s_float = s.cast(pl.Float64, strict=False)

            # Determine if the column can be represented as integers:
            # all non-null values have zero fractional part.
            if s_float.null_count() == 0:
                # Check integrality on non-null values
                non_null = s_float.drop_nulls()
                can_be_int = ((non_null % 1) == 0).all()
            else:
                can_be_int = False

            if can_be_int:
                converted[col] = s_float.cast(pl.Int64)
            else:
                converted[col] = s
        except Exception:
            # If anything goes wrong, leave the column unchanged
            converted[col] = s

    # Rebuild a DataFrame with the same column order
    return pl.DataFrame({col: converted[col] for col in df.columns})


def define_latex_table(args, name, df, remove_header=True, group_column=None):
    """Write a LaTeX table body file from a DataFrame.

    Args:
        group_column: 0-based column index to group by.  When set, the column
            value is shown only on the first row of each group and an
            ``\\addlinespace`` is inserted between groups.
    """
    if args.latex_output is not None:
        with open(Path(args.latex_output, name + ".tex"), "w") as f:
            if isinstance(df, pl.DataFrame):
                converted = convert_to_int_columns(df).to_pandas()
            else:
                converted = df
            latex_table = converted.style.hide(axis="index").format(escape=None).to_latex(siunitx=True, hrules=True)
            middle = latex_table.split(r"\midrule")[1] if remove_header else latex_table.split(r"\toprule")[1]
            table_body = (
                middle
                .split(r"\bottomrule")[0]
                .strip()
            )

            if group_column is not None:
                table_body = _group_rows(table_body, group_column)

            f.write(table_body)
            f.write("\n")


def _group_rows(table_body, col_idx):
    """Post-process a LaTeX table body to group consecutive identical values.

    For the column at *col_idx*, repeated values are blanked out and an
    ``\\addlinespace`` separator is inserted between groups.
    """
    lines = [l.strip() for l in table_body.split("\n") if l.strip()]
    out = []
    prev_value = None
    for line in lines:
        # Detect and preserve the \\ row terminator
        suffix = ""
        stripped = line
        if stripped.rstrip().endswith("\\\\"):
            suffix = " \\\\"
            stripped = stripped.rstrip()[:-2].rstrip()

        cells = stripped.split("&")
        if len(cells) <= col_idx:
            out.append(line)
            continue
        current_value = cells[col_idx].strip()
        if current_value == prev_value:
            cells[col_idx] = " "
        elif prev_value is not None and current_value != "":
            out.append("\\addlinespace")
            prev_value = current_value
        else:
            prev_value = current_value
        out.append(" & ".join(c.strip() for c in cells) + suffix)
    return "\n".join(out)


defines_cleared = False


def define_latex_var(args, name, value):
    global defines_cleared
    if args.latex_output is not None:
        defines = Path(args.latex_output, "defines.tex")
        if not defines_cleared:
            open(defines, "w").close()
            defines_cleared = True

        with open(defines, "a") as f:
            if isinstance(value, float):
                value = f"\\num{{{value}}}"
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")


def load_name_map(name_map_path_str: str | None) -> dict | None:
    """Loads a JSON file mapping experiment folder names to display names."""
    if not name_map_path_str:
        return None

    name_map_path = Path(name_map_path_str)
    if not name_map_path.is_file():
        logging.error(f"Name map file not found: {name_map_path}")
        sys.exit(1)

    try:
        with open(name_map_path, "r") as f:
            name_map = json.load(f)
        logging.info(f"Loaded experiment name map from {name_map_path}")
        return name_map
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from name map file {name_map_path}: {e}")
        sys.exit(1)