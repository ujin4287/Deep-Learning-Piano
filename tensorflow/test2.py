import os

dir = 'C:/Users/ujin4/PycharmProjects/untitled/sample'

file_name_list = []

for pack in os.walk(dir):

    for i in pack[2]:
        file_name_list.append(i)
    print(file_name_list)