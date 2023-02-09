import csv
import sys
import os
import json
import math
import pandas as pd

import sys

sys.path.append("..")

# Network_Root_Path = "./network/4Glogs"
# Network_Root_Path = "./network/generate"
Video_Root_Path = "./data_trace/video/manifest"
User_Root_Path = "./data_trace/video/pose_trace"

with open("test_data_info.csv", 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_head = ["network", "video", "user"]
    csv_writer.writerow(csv_head)

config_file = './headset/headset_config.json'
with open(config_file) as file:
    obj = json.load(file)
    tiles_x = int(obj['tiles_x'])
    tiles_y = int(obj['tiles_y'])
    fov_x_degrees = int(obj['fov_x_degrees'])
    fov_y_degrees = int(obj['fov_y_degrees'])
    segment_ms = int(obj['segment_ms'])

    if tiles_x < 1 or tiles_y < 1:
        print('Headset configuration "%s" has bad "tiles_x" or "tiles_y".' % config_file, file=sys.stderr)
        sys.exit(1)

    if tiles_x == 1 and tiles_y == 1:
        bit_tile_0 = 1
        coords = [(0, 0)]
    else:
        if obj['bit_1_is_tile_0']:
            bit_tile_0 = 1
        else:
            bit_tile_0 = 1 << (tiles_x * tiles_y - 1)
        x_begin = int(obj['tile_0']['x'])
        y_begin = int(obj['tile_0']['y'])
        x_step = int(obj['tile_1']['x']) - x_begin
        y_step = int(obj['tile_1']['y']) - y_begin

        if ((x_step == 0 and y_step == 0) or
                (x_step != 0 and y_step != 0) or
                (tiles_x == 1 and x_step != 0) or
                (tiles_y == 1 and y_step != 0) or
                (x_begin != 0 and x_begin != tiles_x - 1) or
                (y_begin != 0 and y_begin != tiles_y - 1) or
                (y_step == 0 and x_begin == 0 and x_step != 1) or
                (y_step == 0 and x_begin != 0 and x_step != -1) or
                (x_step == 0 and y_begin == 0 and y_step != 1) or
                (x_step == 0 and y_begin != 0 and y_step != -1)):
            print('Headset configuration "%s" has bad "tile_0" or "tile_1".' % config_file, file=sys.stderr)
            sys.exit(1)

        if x_begin == 0:
            xr = range(tiles_x)
        else:
            xr = range(tiles_x - 1, -1, -1)

        if y_begin == 0:
            yr = range(tiles_y)
        else:
            xy = range(tiles_y - 1, -1, -1)

        coords = []
        bit = bit_tile_0
        if y_step == 0:
            for y in yr:
                for x in xr:
                    coords += [(x, y, bit)]
                    if bit_tile_0 == 1:
                        bit <<= 1
                    else:
                        bit >>= 1
        else:
            for x in xr:
                for y in yr:
                    coords += [(x, y, bit)]
                    if bit_tile_0 == 1:
                        bit <<= 1
                    else:
                        bit >>= 1

    tile_sequence = tuple(coords)

    width_x = tiles_x * fov_x_degrees / 360
    width_y = tiles_y * fov_y_degrees / 180

    view_format = '%%0%dX' % ((tiles_x * tiles_y + 3) // 4)


def Pose2VideoXY(pose):
    '''
    将pose四元组转换为视频xy轴的坐标表示
    tiles_x: x轴上tile个数
    tiles_y: y轴上tile个数
    '''
    # 坐标转换
    (qx, qy, qz, qw) = pose
    x = 2 * qx * qz + 2 * qy * qw
    y = 2 * qy * qz - 2 * qx * qw
    z = 1 - 2 * qx * qx - 2 * qy * qy

    video_x = math.atan2(-z, x)
    if video_x < 0:
        video_x += 2 * math.pi
    video_x /= (2 * math.pi)  # 举例：video_x的范围就是0~1
    video_y = math.atan2(math.sqrt(x * x + z * z), y) / math.pi

    return (video_x, video_y)


def take_num(filename):
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ret = ""
    for ele in filename:
        if ele in num_list:
            ret += ele
    return int(ret)


def get_trace_file(Network_Root_Path, network_trace_id, video_trace_id, user_trace_id):
    '''
    获取trace文件
    '''
    # network
    network_files = os.listdir(Network_Root_Path)
    network_files.sort(key=take_num)
    network_trace = os.path.join(Network_Root_Path, network_files[network_trace_id])
    # video
    video_files = os.listdir(Video_Root_Path)
    video_files.sort(key=take_num)
    video_trace = os.path.join(Video_Root_Path, video_files[video_trace_id])
    # user
    user_files = os.listdir(User_Root_Path + '/video_' + str(video_trace_id))
    user_files.sort(key=take_num)
    user_trace = os.path.join(User_Root_Path + '/video_' + str(video_trace_id), user_files[user_trace_id])
    # print(network_trace)
    # print(video_trace)
    # print(user_trace)
    file_path = "test_data_info.csv"
    data_row = [network_trace, video_trace, user_trace]
    with open(file_path, 'a+', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data_row)
    return network_trace, video_trace, user_trace


def get_tiles_in_viewport(x_pred, y_pred):
    '''
    根据视角预测结果返回在视窗内的tiles
    '''
    video_x = x_pred * tiles_x
    video_y = y_pred * tiles_y

    left = video_x - width_x / 2
    right = video_x + width_x / 2
    top = video_y - width_y / 2
    bottom = video_y + width_y / 2

    if left < 0:
        left += tiles_x
    if right >= tiles_x:
        right -= tiles_x
    wrap = left > right

    # 和仿真器保持一致，在y轴方向进行截断
    if top < 0:
        top = 0
    if bottom > tiles_y:
        bottom = tiles_y

    tiles = 0
    tiles_in_viewport = []
    for (tx, ty, bit) in tile_sequence:
        if ty + 1 >= top and ty <= bottom:
            if not wrap:
                if tx + 1 >= left and tx <= right:
                    tiles_in_viewport.append(ty * tiles_x + tx)
                    tiles |= bit
            else:
                if tx + 1 >= left or tx <= right:
                    tiles_in_viewport.append(ty * tiles_x + tx)
                    tiles |= bit
    tiles_in_viewport.sort()
    return tiles_in_viewport


def calculate_viewing_proportion(video_x, video_y, tile_idx):
    video_x = video_x * tiles_x
    video_y = video_y * tiles_y

    left = video_x - width_x / 2
    right = video_x + width_x / 2
    top = video_y - width_y / 2
    bottom = video_y + width_y / 2

    if left < 0:
        left += tiles_x
    if right >= tiles_x:
        right -= tiles_x
    wrap = left > right  # 翻转

    # 计算tile坐标()
    tx = tile_idx % tiles_x
    ty = tile_idx // tiles_x

    # 计算tile对视窗的覆盖面积
    height = 0
    if top <= ty <= bottom - 1:
        height = 1
    elif ty < top < ty + 1:
        height = ty + 1 - top
    elif ty < bottom < ty + 1:
        height = bottom - ty

    width = 0
    if not wrap:
        if left <= tx <= right - 1:
            width = 1
        elif tx < left < tx + 1:
            width = tx + 1 - left
        elif tx < right < tx + 1:
            width = right - tx
    else:
        if tx + 1 <= right or tx >= left:
            width = 1
        elif tx < right < tx + 1:
            width = right - tx
        elif tx < left < tx + 1:
            width = tx + 1 - left
    assert width >= 0
    assert height >= 0

    area = width * height
    proportion = area / 1.0

    return proportion


def print_metrics(metrics):
    ''' ====== 打印各项指标 =================================
       [0]score [1]qoe [2]quality [3]stall_time [4]var_space
       [5]var_time [6]bandwidth_usage [7]bandwidth_wastage'''
    print("------------------------------------------------------------------")
    print('Score: {:.2f}'.format(metrics[0]))
    print('QoE: {:.2f}\t\tbandwidth_usage: {:.2f}'.format(metrics[1], metrics[6]))
    print('Quality: {:.2f}\tStall time: {:.2f}\t'.format(metrics[2], metrics[3]))
    print('Oscillation in space: {:.2f}\tOscillation in time: {:.2f}'.format(metrics[4], metrics[5]))
    wastage_ratio = metrics[7] / metrics[6]
    print('Bandwidth wastage: {:.2f}'.format(metrics[7]))
    print('Wastage ratio: {:.2f}\n'.format(wastage_ratio))


def print_to_csv(file_path, metrics):
    ''' ====== 结果打印至csv文件 =============================
       [0]score [1]qoe [2]quality [3]stall_time [4]var_space
       [5]var_time [6]bandwidth_usage [7]bandwidth_wastage'''
    data_row = []
    data_row.append(metrics[2])  # Quailty
    data_row.append(metrics[4])  # Var_space
    data_row.append(metrics[5])  # Var_time
    data_row.append(metrics[3])  # Stall_time
    data_row.append(metrics[1])  # QoE
    wastage_ratio = metrics[7] / metrics[6]
    data_row.append(wastage_ratio)  # Wastage ratio
    data_row.append(metrics[0])  # Score

    with open(file_path, 'a+', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data_row)


def create_csv(file_path):
    with open(file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_head = ["Quailty", "Var_space", "Var_time", "Stall_time", "QoE", "Wastage ratio", "Score"]
        csv_writer.writerow(csv_head)


if __name__ == '__main__':
    proportion = calculate_viewing_proportion(0, 0, 8)
    print(proportion)
