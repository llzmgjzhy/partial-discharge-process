# from mysql_connect import DatabaseConnection
import json
import struct
import pandas as pd
from mysql.connector import Error
from save_mysql import get_data_storage_method
import binascii
import numpy as np
import struct


class MyHttp:
    def __init__(self):
        self.filename = None
        self.data = []

    def check_single_info(self, data):
        with open("HEADFILE.json", "r") as file:
            table_structure = json.load(file)
        instrument_serial_number = data[
            table_structure["HEAD_FILE"]["instrument_serial_number"]["index"][
                0
            ] : table_structure["HEAD_FILE"]["instrument_serial_number"]["index"][1]
            + 1
        ]

        return instrument_serial_number

    def process_complete_data(self, data, data_info):
        # save_data_info(data_info)
        self.filename = data_info["file_name"]
        # 处理完整的数据
        print("处理完整的数据，数据大小:", len(data))
        # 解析头部信息
        map_quantity = int.from_bytes(data[286:288], "little")  # 图谱数量
        print("图谱数量：", map_quantity)
        # 初始化指针，跳过头部
        pointer = 512
        # head_file_data = data[:pointer]
        # self.save_head_file_info_mysql(head_file_data)  # 保存头文件信息

        # 实例化对象
        HF_map = High_frequency_map()
        UHF_map = Ultra_High_frequency_map()

        # 循环解析每个图谱数据
        for _ in range(map_quantity):
            if pointer + 4 > len(data):
                raise ValueError("数据长度不足以包含图谱大小信息")

            # 读取图谱类型
            map_type = data[pointer]
            print("图谱类型:", hex(map_type))

            # 读取图谱大小
            map_size = int.from_bytes(data[pointer + 1 : pointer + 5], "little")

            if pointer + map_size > len(data):
                raise ValueError("数据长度不足以包含完整的图谱数据")

            # 读取图谱数据
            map_data = data[pointer : pointer + map_size]
            pointer += map_size  # 移动指针
            # 根据图谱类型处理数据
            if map_type == 0x11:
                # 处理高频PRPD图
                continue
                data_prpd = HF_map.process_hf_prpd_map(self.filename, map_data)
            elif map_type == 0x12:
                # 处理高频PRPS图
                continue
                HF_map.process_hf_prps_map(self.filename, map_data)
            elif map_type == 0x13:
                # 处理高频脉冲波形图
                continue
                HF_map.process_hf_pulse_waveform_map(self.filename, map_data)
            elif map_type == 0x21:
                # handle uhf prpd map
                prpd_data = UHF_map.process_uhf_prpd_map(self.filename, map_data)
                self.data.append(prpd_data)
            elif map_type == 0x22:
                # handle uhf prps map
                continue
                prpd_data = UHF_map.process_uhf_prps_map(self.filename, map_data)
                self.data.append(prpd_data)
            else:
                print("未知的图谱类型")

        print("所有图谱数据解析完毕")


class High_frequency_map:
    def __init__(self):
        self.data_buffer = bytearray()

    def process_hf_prpd_map(self, filename, map_data):
        # 处理高频PRPD图的函数
        print("处理高频PRPD图，数据长度:", len(map_data))
        data_prpd = self.save_hf_prpd_info_mysql(filename, map_data)
        return data_prpd

    def process_hf_prps_map(self, filename, map_data):
        # 处理高频PRPS图的函数
        print("处理高频PRPS图，数据长度:", len(map_data))
        self.save_hf_prps_info_mysql(filename, map_data)

    def process_hf_pulse_waveform_map(self, filename, map_data):
        # 处理高频PRPS图的函数
        print("处理高频PRPS图，数据长度:", len(map_data))
        self.save_hf_pulse_waveform_info_mysql(filename, map_data)

    def save_hf_prpd_info_mysql(self, filename, data):
        print("读取prpd数据")
        # 建立数据表
        t = data[336:337]  # 存储数据类型t
        d, k = get_data_storage_method(t)  # 字节数
        print("k的值为：{}".format(k))
        m = int.from_bytes(data[355:359], "little")  # 相位窗数m
        n = int.from_bytes(data[359:363], "little")  # 量化幅值n

        # 数据解析和插入
        parsed_data = []
        data_start_index = 512  # 假设数据从此索引开始
        for _ in range(m):
            # 读取可变部分
            variable_part = data[data_start_index : data_start_index + n * k]
            # 将字节的可变部分转换为整数
            parsed_data.append(parse_data(variable_part, d, k))

            # 更新下一组数据的起始索引
            data_start_index += k * n
        all_data = np.array(parsed_data)
        all_data = np.transpose(all_data)
        parsed_data = all_data.tolist()
        return parsed_data

    def save_hf_prps_info_mysql(self, filename, data):
        print("读取prps数据")
        t = data[336:337]  # 存储数据类型t
        d, k = get_data_storage_method(t)  # 字节数
        print("k的值为：{}".format(k))
        m = int.from_bytes(data[355:359], "little")  # 相位窗数m
        p = int.from_bytes(data[363:367], "little")  # 工频周期数p
        # 数据解析和插入
        parsed_data = []
        data_start_index = 512  # 假设数据从此索引开始
        for _ in range(p):
            # 读取可变部分
            variable_part = data[data_start_index : data_start_index + m * k]

            # 将字节的可变部分转换为整数
            parsed_data.extend(parse_data(variable_part, d, k))

            # 更新下一组数据的起始索引
            data_start_index += k * m

        return parsed_data


class Ultra_High_frequency_map:
    """class UHF"""

    def __init__(self):
        self.data_buffer = bytearray()

    def process_uhf_prpd_map(self, filename, map_data):
        # 处理高频PRPD图的函数
        print("处理特高频PRPD图，数据长度:", len(map_data))
        data_prpd = self.save_uhf_prpd_info_mysql(filename, map_data)
        return data_prpd

    def process_uhf_prps_map(self, filename, map_data):
        # 处理高频PRPS图的函数
        print("处理特高频PRPS图，数据长度:", len(map_data))
        data_prps = self.save_uhf_prps_info_mysql(filename, map_data)
        return data_prps

    def process_uhf_peak_statistics_map(self, filename, map_data):
        # 处理高频PRPS图的函数
        print("处理特高频峰值统计图，数据长度:", len(map_data))
        self.save_uhf_pulse_waveform_info_mysql(filename, map_data)

    def save_uhf_prpd_info_mysql(self, filename, data):
        print("读取prpd数据")
        # 建立数据表
        t = data[336:337]  # 存储数据类型t
        d, k = get_data_storage_method(t)  # 字节数
        print("k的值为：{}".format(k))
        m = int.from_bytes(data[355:359], "little")  # 相位窗数m
        n = int.from_bytes(data[359:363], "little")  # 量化幅值n

        # 数据解析和插入
        parsed_data = []
        data_start_index = 512  # 假设数据从此索引开始
        for _ in range(m):
            # 读取可变部分
            variable_part = data[data_start_index : data_start_index + n * k]
            # 将字节的可变部分转换为整数
            parsed_data.append(parse_data(variable_part, d, k))

            # 更新下一组数据的起始索引
            data_start_index += k * n
        all_data = np.array(parsed_data)
        all_data = np.transpose(all_data)
        parsed_data = all_data.tolist()
        return parsed_data

    def save_uhf_prps_info_mysql(self, filename, data):
        print("读取prps数据")
        t = data[336:337]  # 存储数据类型t
        d, k = get_data_storage_method(t)  # 字节数
        print("k的值为：{}".format(k))
        m = int.from_bytes(data[355:359], "little")  # 相位窗数m
        p = int.from_bytes(data[363:367], "little")  # 工频周期数p
        # 数据解析和插入
        parsed_data = []
        data_start_index = 512  # 假设数据从此索引开始
        for _ in range(p):
            # 读取可变部分
            variable_part = data[data_start_index : data_start_index + m * k]

            # 将字节的可变部分转换为整数
            parsed_data.append(parse_data(variable_part, d, k))

            # 更新下一组数据的起始索引
            data_start_index += k * m

        return parsed_data


def parse_data(variable_part, d, k):
    parsed_data = []
    if d == "float":
        # print("长度为：{}".format(len(variable_part)))
        parsed_data = [
            struct.unpack("<f", variable_part[i : i + k])[0]
            for i in range(0, len(variable_part), k)
        ]
    elif d in ["uint8", "int16", "int32", "int64"]:
        parsed_data = [
            int.from_bytes(variable_part[i : i + k], "little")
            for i in range(0, len(variable_part), k)
        ]
    elif d == "double":
        parsed_data = [
            struct.unpack("<d", variable_part[i : i + k])[0]
            for i in range(0, len(variable_part), k)
        ]
    else:
        print("数据类型存在错误")

    return parsed_data
