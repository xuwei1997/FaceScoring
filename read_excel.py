# 数据准备
# 读取excel，转化为dict
# dict格式 {'照片名.jpg':分数}

import xlrd

def get_data(filename, sheet):
    dict = {}
    dir_case = "E:\\daxue\\graduation\\" + filename + '.xlsx'
    data = xlrd.open_workbook(dir_case)
    #table = data.sheets()[sheetnum]
    table = data.sheet_by_name(sheet)
    #table = data.sheet_by_index(sheet)
    nor = table.nrows
    nol = table.ncols
    print(nor,nol)
    for i in range(0,5502):
        key=table.cell(i,0).value
        val=table.cell(i,1).value
        dict[key]=val
    return dict





if __name__ == '__main__':
    k=get_data('data', 'data_sheet')
    print(k)
    print(k['AF1816.jpg'])
