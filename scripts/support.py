import os
import re
import warnings
from operator import add

import numpy as np
import pandas as pd
import timeit

# Отключаем warnings
warnings.simplefilter("ignore")

"""
    Методы:
    -----------
    - load_Rosstat_separated_data(**path_and_sheetnames) - чтение данных из таблиц Росстата. Таблицы отечественного
    выпуска и импорта находятся на разных страницах, данные за разные годы лежат в разных файлах.
    
    - pack_name(str) - приклеивает к строке годы, к которым относится содержание строки, и если было указано, то год,
    для которого были пересчитаны цены в таблице
    - save_to_excel(file_name, rounding="%.3f", **tables) - сохраняет полученные таблицы в выбранный excel-файл
"""


def load_Rosstat_separated_data(**path_and_sheetnames):
    """
    Чтение данных из таблиц Росстата. Таблицы отечественного выпуска и импорта находятся на разных страницах,
    данные за разные годы лежат в разных файлах.

    Parameters
    ----------
    path_and_sheetnames : dictionary with string as key and list as value
        в качестве ключа используется путь к excel-файлу, а в качестве значения - список номеов\названий страниц в
        excel-файле

    Returns
    -------
    codes : pandas.Series
        коды отраслей
    years : list
        годы за которые приведены таблицы
    df_d : list of pandas dataframes
        таблицы отечественного выпуска
    df_m : list of pandas dataframes
        таблицы отечественного выпуска
    """

    # Расположение таблицы и столбцов\строк с названиями в ней
    vertical_table_position = slice(3, 62)  # положение и размеры таблицы по вертикали
    horizontal_table_position = slice(3, 69)  # положение и размеры таблицы по горизонтали
    industries_part_position = slice(3, 62)  # положение и размеры части таблицы с промежуточным потреблением по
    # горизонтали
    codes_position = 1  # номер строки в таблице с кодами отраслей
    columns_names_position = 0  # номер строки в таблице с названиями колонок
    rows_names_position = 2  # номер столбца в таблице с названиями строк

    years = []  # годы за которые приведены таблицы
    df_d = []  # таблицы отечественного выпуска
    df_m = []  # таблицы отечественного выпуска

    for path, sheetnames in path_and_sheetnames.items():
        file = pd.ExcelFile(path)
        for i, sheetname in enumerate(sheetnames):
            df1 = pd.read_excel(file, sheet_name=sheetname)

            # Получаем имена столбцов и строк
            rows = df1.iloc[vertical_table_position, rows_names_position]
            rows.name = ""
            columns = df1.iloc[columns_names_position, horizontal_table_position]
            columns.name = df1.columns[0]

            # Получаем из таблицы коды отраслей (одинаковые по вертикали и горизонтали)
            codes = df1.iloc[codes_position, industries_part_position]

            # Сохраняем обрезанную версию таблицы
            df = df1.iloc[vertical_table_position, horizontal_table_position]
            df.columns = columns
            df.index = rows
            df.name = columns.name

            # Делим таблицы на отечественный выпуск и импорт
            if not i:
                df_d.append(df)
            else:
                df_m.append(df)

        table_format = "Rosstat"
        print("Обрабатываем данные из таблицы в формате " + table_format + " \"" + columns.name + "\"")
        years.append(re.search("\d+", columns.name).group(0))
    years = years[0] + "-" + years[1]
    return codes, years, df_d, df_m


def pack_name(str, years, prices_in):
    """
    Приклеивает к строке годы, к которым относится содержание строки, и если было указано, то год, для которого были
    пересчитаны цены в таблице

    str : string
        строка-название таблицы\страницы файла
    """
    return str + " за " + years + "гг(" + prices_in + ")"

def save_to_excel(file_name, years, codes, rounding="%.3f", table_format = "Rosstat",  **tables):
    """
    Cохраняет полученные таблицы в выбранный excel-файл постранично

    Parameters
    ----------
    file_name: string
        имя файла
    rounding: string
        Формат округления чисел в сохраняемых таблицах.
        по умолчанию - "%.3f"(до 3 знаков после запятой),
        для результатов в процентах - "%.1f"(до 1 знака после запятой)
    tables: dictionary
        Словарь из названий таблиц и самих таблиц

    """
    #wiod16_flag = "w16" if self.table_format == "WIOD16" else ""
    writer = pd.ExcelWriter("./results/" + years + "/" + table_format + "/" + file_name, #+ wiod16_flag + file_name,
                            engine='xlsxwriter')
    workbook = writer.book

    for table_name, df in tables.items():
        sheet_name = df.columns.name

        df.to_excel(writer, sheet_name=sheet_name, float_format=rounding, startrow=1, startcol=2,
                    header=False, index=False)

        worksheet = writer.sheets[sheet_name]
        worksheet.set_zoom(80)
        worksheet.set_column(2, len(df.columns) + 1, 23)

        # Add a header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#D7E4BC',
            'border': 1})

        # Add total row formating
        if any(df.index.values == 'Total'):
            total_format = workbook.add_format({'bold': True, 'fg_color': '#ffb74d', 'border': 1})
            worksheet.write_row(np.shape(df)[0], 2, np.array(df.loc['Total', :]), total_format)

        # Add codes & index
        if len(df.index) > len(codes):
            codes2 = pd.concat([pd.Series(['']), pd.DataFrame(codes.tolist())])
            worksheet.write_column('A1', codes2, header_format)
        worksheet.write_column('B1', np.insert(df.index.values, 0, ''), header_format)
        worksheet.set_column('B:B', 45)

        # Add header
        columns = np.insert(df.columns.values, 0, table_name)
        for col_num, value in enumerate(columns):
            worksheet.write(0, col_num + 1, value, header_format)
        worksheet.set_row(0, 75)

    writer.save()
    workbook.close()
