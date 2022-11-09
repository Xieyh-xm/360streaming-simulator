# Copyright (c) 2020, authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
包含QOE计算的Sabre360，且仅在segment中至少有一个tile时才播放
'''

import copy
from utils import Pose2VideoXY, calculate_viewing_proportion
# Units used throughout:
#     size     : bits
#     time     : ms
#     size/time: bits/ms = kbit/s


# Main TODOs:
# 1. Update ABR algorithm
# 2. Test view prediction algorithm
# 3. Support download pipelining for mutiple GETs per segment with non-zero RTT


import json
import math
import sys
from collections import namedtuple

from headset import headset
from abr.baseline import BaselineAbr, TrivialThroughputAbr, ThreeSixtyAbr
from abr.myABR import TestAbr
from view_prediction import TestPrediction

g_debug_cycle = 0
TILES_X = 8
TILES_Y = 8


def load_json(path):
    with open(path) as file:
        obj = json.load(file)
    return obj


# segment_duration in ms e.g. 3000
# tiles is number of tiles e.g. 12. Note that tile layout such as 4x3 does not matter here
# bitrates is list of bitrates per tile e.g. [100, 1000] for 100 kbps, 1 Mbps per tile (12 tiles give 1.2, 12 Mbps)
# utilities is list of relative utility per bitrate
# segments[index][tile][quality] is a size in bits
ManifestInfo = namedtuple('ManifestInfo', 'segment_duration tiles bitrates utilities segments')

# A network period lasts "time" ms, provides "bandwidth" kbps, and has "latency" RTT delay
NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

# play_time: video presentation time for given head pose
# pose: the pose information (TODO: clarify)
PoseInformation = namedtuple('PoseInformation', 'play_time pose')

# index tile quality: what download is in progress
# size: the size in bits of the download when ready
# downloaded: how many bits have been received by now
# time: total time from request sent until DownloadProgress measured
# time_to_first_bit: time from request sent to first bit (see "latency" in NetworkPeriod)
# abandon: if download was abandoned, then abandon contains information about new action, else abandon is None
DownloadProgress = namedtuple('DownloadProgress', 'segment tile quality size downloaded time time_to_first_bit abandon')


def str_download_progress(self):
    if self.abandon is None:
        abandon = ''
    else:
        abandon = ' abandon_to:(%s)' % str(self.abandon)
    t = self.time / 1000
    ttfb = self.time_to_first_bit / 1000  # s
    tffb = (self.time - self.time_to_first_bit) / 1000  # s
    return ('segment:%d tile:%d quality:%d %d/%dbits %.3f+%.3f=%.3f%ss' %
            (self.segment, self.tile, self.quality, self.downloaded, self.size, ttfb, tffb, t, abandon))


DownloadProgress.__str__ = str_download_progress


class TiledBuffer:

    def __init__(self, segment_duration, tiles):

        self.segment_duration = segment_duration
        self.tiles = tiles
        self.buffers = []
        self.played_segments = 0
        self.played_segment_partial = 0  # measured in ms
        self.quality_per_segment = {}  # 根据实际的视角计算平均视窗质量
        self.quality_var_time_dict = {}
        self.quality_var_space_dict = {}

    def __str__(self):
        segment0 = self.played_segments
        ret = ('playhead:%.3f first_segment:%d first_segment_offset:%.3fs segment_depth:%d contents:' %
               (self.get_play_head() / 1000, self.played_segments, self.played_segment_partial / 1000,
                len(self.buffers)))
        for i in range(len(self.buffers)):
            ret += ' segment=%d: (' % (segment0 + i)
            delim = ''
            for t in range(self.tiles):
                q = self.buffers[i][t]
                ret += delim
                delim = ', '
                if q is None:
                    ret += '-'
                else:
                    ret += str(q)
            ret += ')'
        return ret

    def get_played_segments(self):
        return self.played_segments

    def get_played_segment_partial(self):
        return self.played_segment_partial

    def get_buffer_head(self):
        return self.played_segments * self.segment_duration

    def get_play_head(self):
        '''
        :return: 实时播放位置
        '''
        return self.played_segments * self.segment_duration + self.played_segment_partial

    def get_buffer_depth(self):
        '''
        :return: 缓冲区长度
        '''
        return len(self.buffers)

    def get_buffer_element(self, segment, tile):
        ''' 返回buffer内指定tile的码率 '''
        index = segment - self.played_segments
        if not 0 <= index < len(self.buffers):
            return None
        return self.buffers[index][tile]

    def get_buffer_contents(self, segment):
        index = segment - self.played_segments
        assert index >= 0
        if index >= len(self.buffers):
            return None
        return self.buffers[index]

    def get_segment_quality(self, segment_idx):
        ''' 按segment ID提取视窗内画面平均quality '''
        if segment_idx not in self.quality_per_segment:
            return None
        return self.quality_per_segment[segment_idx]

    def add_segment_quality(self, segment_idx, delta_quality):
        ''' 按segment ID保存视窗内画面总quality '''
        if segment_idx in self.quality_per_segment:
            self.quality_per_segment[segment_idx] += delta_quality
        else:
            self.quality_per_segment[segment_idx] = delta_quality
        return

    def get_quality_var_time(self, segment_idx):
        ''' 读取时间上的不平滑 '''
        if segment_idx in self.quality_var_time_dict:
            return self.quality_var_time_dict[segment_idx]
        else:
            return 0

    def save_quality_var_time(self, segment_idx, var):
        ''' 保存时间上的不平滑 '''
        self.quality_var_time_dict[segment_idx] = var

    def get_quality_var_space(self, segment_idx):
        ''' 读取空间上的不平滑 '''
        if segment_idx in self.quality_var_space_dict:
            return self.quality_var_space_dict[segment_idx]
        else:
            return 0

    def save_quality_var_space(self, segment_idx, var):
        ''' 保存空间上的不平滑 '''
        self.quality_var_space_dict[segment_idx] = var

    def put_in_buffer(self, segment_index, tile_index, quality):
        segment = segment_index - self.played_segments
        if segment < 0:
            # allow case when (segment == 0 and self.played_segment_partial > 0) here,
            # but be careful elsewhere when updating a segment while playing it
            return segment
        grow = segment + 1 - len(self.buffers)
        if grow > 0:
            self.buffers += [[None] * self.tiles for g in range(grow)]
        self.buffers[segment][tile_index] = quality
        return segment

    def play_out_buffer(self, play_time):
        ''' 更新播放后buffer '''
        self.played_segment_partial += play_time
        if self.played_segment_partial >= self.segment_duration:
            discard = int(self.played_segment_partial // self.segment_duration)
            del self.buffers[:discard]
            self.played_segments += discard
            self.played_segment_partial %= self.segment_duration


class SessionInfo:

    def __init__(self, manifest, buffer, buffer_size):
        self.manifest = manifest
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.wall_time = 0
        self.presentation_time = 0

    def set_throughput_estimator(self, throughput_estimator):
        self.throughput_estimator = throughput_estimator

    def set_viewport_predictor(self, viewport_predictor):
        self.viewport_predictor = viewport_predictor

    def set_user_model(self, user_model):
        self.user_model = user_model

    def set_log_file(self, log_file):
        self.log_file = log_file

    def get_manifest(self):
        return self.manifest

    def get_buffer(self) -> TiledBuffer:
        return self.buffer

    def get_user_model(self):
        return self.user_model

    def get_throughput_estimator(self):
        return self.throughput_estimator

    def get_viewport_predictor(self):
        return self.viewport_predictor

    def get_log_file(self):
        return self.log_file

    def advance_wall_time(self, t):
        self.wall_time += t

    def get_wall_time(self):
        return self.wall_time


class SessionEvents:

    def __init__(self):
        self.play_handlers = []
        self.stall_handlers = []
        self.network_delay_handlers = []
        self.pose_handlers = []

    def add_handler(self, list_of_handlers, handler):
        if handler not in list_of_handlers:
            list_of_handlers += [handler]

    def remove_handler(self, list_of_handlers, handler):
        if handler in list_of_handlers:
            list_of_handlers.remove(handler)

    def trigger_event_0(self, list_of_handlers):
        for handler in list_of_handlers:
            handler()

    def trigger_event_1(self, list_of_handlers, value1):
        for handler in list_of_handlers:
            handler(value1)

    def add_play_handler(self, handler):
        self.add_handler(self.play_handlers, handler)

    def remove_play_handler(self, handler):
        self.remove_handler(self.play_handlers, handler)

    def trigger_play_event(self, time):
        self.trigger_event_1(self.play_handlers, time)

    def add_stall_handler(self, handler):
        self.add_handler(self.stall_handlers, handler)

    def remove_stall_handler(self, handler):
        self.remove_handler(self.stall_handlers, handler)

    def trigger_stall_event(self, time):
        self.trigger_event_1(self.stall_handlers, time)

    def add_network_delay_handler(self, handler):
        self.add_handler(self.network_delay_handlers, handler)

    def remove_network_delay_handler(self, handler):
        self.remove_handler(self.network_delay_handlers, handler)

    def trigger_network_delay_event(self, time):
        self.trigger_event_1(self.network_delay_handlers, time)

    def add_pose_handler(self, handler):
        self.add_handler(self.pose_handlers, handler)

    def remove_pose_handler(self, handler):
        self.remove_handler(self.pose_handlers, handler)

    def trigger_pose_event(self, play_time, pose):
        ''' 调用视角预测算法 '''
        self.trigger_event_1(self.pose_handlers, (play_time, pose))


class LogFileForTraining:
    def __init__(self, session_info, path):
        self.session_info = session_info

    def log_str(self, s):
        pass

    def log_new_cycle(self, index):
        pass

    def log_tput(self, tput):
        pass

    def log_view(self, view):
        pass

    def log_abr(self, action, from_abandoned=False):
        pass

    def log_abr_replace(self, action, old_quality, from_abandoned=False):
        pass

    def log_pose(self, pose):
        pass

    def log_play(self, stall_time, is_startup):
        pass

    def log_stall(self):
        pass

    def log_delay(self, delay):
        pass

    def log_download(self, progress):
        pass

    def log_check_abandon(self, progress):
        pass

    def log_check_abandon_verdict(self, action):
        pass

    def log_abandoned(self, progress):
        pass

    def log_buffer(self, buffer):
        pass

    def log_rendering(self, segment, tiles):
        pass


class LogFile:
    def __init__(self, session_info, path):
        self.session_info = session_info
        self.fo = open(path, 'w')

    def log_str(self, s):
        self.fo.write('[%.3f] [%.3f] %s\n' % (self.session_info.get_wall_time() / 1000,
                                              self.session_info.get_buffer().get_play_head() / 1000,
                                              s))
        self.fo.flush()

    def log_new_cycle(self, index):
        self.log_str('new cycle: %d' % index)

    def log_tput(self, tput):
        self.log_str('throughput: %.0f kbps' % tput)

    def log_view(self, view):
        self.log_str('tile prediction: %s' % view)

    def log_abr(self, action, from_abandoned=False):
        if not from_abandoned:
            self.log_str('action: %s' % str(action))
        else:
            self.log_str('action (from abandon response): %s' % str(action))

    def log_abr_replace(self, action, old_quality, from_abandoned=False):
        if not from_abandoned:
            self.log_str('action: %s (replace quality:%d)' % (str(action), old_quality))
        else:
            self.log_str('action (from abandon response): %s (replace quality:%d)' % (str(action), old_quality))

    def log_pose(self, pose):
        self.log_str('pose: %s: %s' % (str(pose), headset.format_view(headset.get_tiles(pose))))

    def log_play(self, stall_time, is_startup):
        if is_startup:
            description = 'startup time'
        else:
            description = 'rebuffering time'
        self.log_str('play: %s: %.3fs' % (description, stall_time / 1000))

    def log_stall(self):
        self.log_str('stall')

    def log_delay(self, delay):
        self.log_str('delay complete: %.3fs' % (delay / 1000))

    def log_download(self, progress):
        self.log_str('download complete: %s' % str(progress))

    def log_check_abandon(self, progress):
        self.log_str('check abandon: progress: %s' % str(progress))

    def log_check_abandon_verdict(self, action):
        if action is None:
            a = 'No'
        else:
            a = 'Yes (%s)' % str(action)
        self.log_str('check abandon: verdict: %s' % a)

    def log_abandoned(self, progress):
        self.log_str('download abandoned: %s' % str(progress))

    def log_buffer(self, buffer):
        self.log_str('buffer contents: %s' % buffer)

    def log_rendering(self, segment, tiles):
        self.log_str(
            'rendering segment=%d: (%s)' % (segment, ', '.join([(str(t) if t is not None else '-') for t in tiles])))


class VideoModel:

    def __init__(self, manifest):
        self.manifest = manifest

    def get_manifest(self):
        return self.manifest


class HeadsetModel:

    def __init__(self, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.pose_iterator = None
        self.last_pose_info = None
        self.next_pose_info = None
        self.startup_wait = session_info.get_manifest().segment_duration  # todo: start_up的设置
        self.is_playing = False

        log_file = self.session_info.get_log_file()
        tile_sequence = headset.tile_sequence
        if len(tile_sequence) > 0:
            (tx, ty, bit) = tile_sequence[0]
            log_file.log_str('First tile: %s: %s' % (self.describe_x_y(tx, ty), headset.format_view(bit)))
        if len(tile_sequence) > 2:
            (txx, tyy, bit) = tile_sequence[1]
            log_file.log_str('Second tile: %s: %s' % (self.describe_x_y_direction(tx, ty, txx, tyy),
                                                      headset.format_view(bit)))
        if len(tile_sequence) > 1:
            (tx, ty, bit) = tile_sequence[-1]
            log_file.log_str('Last tile: %s: %s' % (self.describe_x_y(tx, ty), headset.format_view(bit)))

    def describe_x_y(self, x, y):
        if y == 0:
            describe_y = 'top'
        elif y == headset.tiles_y - 1:
            describe_y = 'bottom'

        if x == 0:
            describe_x = 'left'
        elif x == headset.tiles_x - 1:
            describe_x = 'right'

        return '%s, %s' % (describe_y, describe_x)

    def describe_x_y_direction(self, x, y, xx, yy):
        # not general, only works in one use case
        assert (y == yy or x == xx)
        assert (y != yy or x != xx)

        if y == 0 and yy == y:
            describe_y = 'top'
        elif y == 0 and yy != y:
            describe_y = 'down'
        elif y == headset.tiles_y - 1 and yy == y:
            describe_y = 'bottom'
        elif y == headset.tiles_y - 1 and yy != y:
            describe_y = 'up'

        if x == 0 and xx == x:
            describe_x = 'left'
        elif x == 0 and xx != x:
            describe_x = 'rightward'
        elif x == headset.tiles_x - 1 and xx == x:
            describe_x = 'right'
        elif x == headset.tiles_x - 1 and xx != x:
            describe_x = 'leftward'

        return '%s, %s' % (describe_y, describe_x)

    # returns a list with a weight between 0.0 and 1.0 for each tile
    def get_tiles_for_pose(self, pose):
        tiles = headset.get_tiles(pose)
        ret = []
        for (tx, ty, bit) in headset.tile_sequence:
            if tiles & bit == 1:
                ret += [1.0]
            else:
                ret += [0.0]
        return ret

    # headset model determines whether to rebuffer or not, and also play rate
    def play_for_time(self, time, buffer, time_is_play_time=False):
        # 先计算卡顿时长
        stall = 0
        buffer_length = buffer.get_buffer_depth() * buffer.segment_duration - buffer.get_played_segment_partial()
        if buffer_length < time:
            stall = time - buffer_length
            assert stall > 0
            time = buffer_length

        wall_time = time
        # if self.startup_wait > 0:
        #     if time_is_play_time:
        #         time += self.startup_wait
        #         wall_time += self.startup_wait
        #
        #     if time >= self.startup_wait:
        #         stall = self.startup_wait
        #         self.startup_wait = 0
        #         time -= stall
        #         self.is_playing = True
        #         self.session_events.trigger_stall_event(stall)
        #     else:
        #         self.startup_wait -= time
        #         self.session_events.trigger_stall_event(time)
        #         return time

        partial_time = 0
        while partial_time < time:
            if self.pose_iterator is None:  # 初始化 self.pose_iterator
                # leave this here - during initialization session_info might not have user model
                self.pose_iterator = self.session_info.get_user_model().get_iterator()
                pose_info = next(self.pose_iterator)
                assert (pose_info is not None)
                self.last_pose_info = pose_info
                self.session_events.trigger_pose_event(self.last_pose_info.play_time, self.last_pose_info.pose)
                self.next_pose_info = next(self.pose_iterator)

            head = buffer.get_play_head()  # 获取实时播放位置

            while self.next_pose_info is not None and head >= self.next_pose_info.play_time:  # 更新用户姿态，调用pose event
                self.last_pose_info = self.next_pose_info
                self.session_events.trigger_pose_event(self.last_pose_info.play_time, self.last_pose_info.pose)
                self.next_pose_info = next(self.pose_iterator)

            skip = time - partial_time
            # make sure we do not skip over segment boundaries:
            remain_in_segment = self.session_info.get_manifest().segment_duration - buffer.get_played_segment_partial()
            if skip > remain_in_segment:
                skip = remain_in_segment
            # make sure we do not skip over pose changes:
            if self.next_pose_info is not None and skip > self.next_pose_info.play_time - head:
                skip = self.next_pose_info.play_time - head

            partial_time += skip
            self.session_events.trigger_play_event(skip)  # 播放skip时间

        return wall_time, stall  # 返回播放时间和卡顿时间


class UserModel:

    def __init__(self, pose_trace):
        self.pose_trace = pose_trace
        self.last_index = 0

    def get_pose(self, time):
        index = self.last_index
        if time < self.pose_trace[index].play_time:
            index = 0
        while index + 1 < len(self.pose_trace) and time >= self.pose_trace[index + 1].play_time:
            index += 1
        if index + 1 < len(self.pose_trace):
            end_time = self.pose_trace[index + 1].play_time
        else:
            end_time = None
        self.last_index = index
        return (self.pose_trace[index].pose, end_time)

    def get_pose_in_qoe(self, start_time, end_time) -> list:
        pose_list = []
        index = 0
        while self.pose_trace[index + 1].play_time <= start_time:
            index += 1
        last_log_time = self.pose_trace[index + 1].play_time
        pose_list.append(self.pose_trace[index + 1].pose)
        # while index + 1 < len(self.pose_trace) and self.pose_trace[index + 1].play_time <= end_time:
        #     index += 1
        #     cur_play_time = self.pose_trace[index].play_time
        #     if cur_play_time - last_log_time > 100.:
        #         pose_list.append(self.pose_trace[index].pose)
        #         last_log_time = cur_play_time
        return pose_list

    def get_iterator(self):
        last_pose_info = None
        for pose_info in self.pose_trace:
            if last_pose_info is None or pose_info.pose != last_pose_info.pose:
                last_pose_info = pose_info
                yield pose_info
        yield None


class NetworkModel:
    min_progress_size = 12000
    min_progress_time = 50

    def __init__(self, network_trace):

        self.network_total_time = 0
        self.trace = network_trace
        self.index = -1
        self.time_to_next = 0
        self.next_network_period()

    def next_network_period(self):
        self.index += 1
        if self.index == len(self.trace):
            self.index = 0
        self.time_to_next = self.trace[self.index].time

    # return delay time
    def do_latency_delay(self, delay_units):

        total_delay = 0
        while delay_units > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_delay += time
                self.network_total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:
                # time > self.time_to_next implies current_latency > 0
                total_delay += self.time_to_next
                self.network_total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_network_period()
        return total_delay

    # return download time
    def do_download(self, size):
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.trace[self.index].bandwidth
            if size <= self.time_to_next * current_bandwidth:
                # current_bandwidth > 0
                time = size / current_bandwidth
                total_download_time += time
                self.network_total_time += time
                self.time_to_next -= time
                size = 0
            else:
                total_download_time += self.time_to_next
                self.network_total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_network_period()
        return total_download_time

    def do_minimal_latency_delay(self, delay_units, min_time):
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
                self.network_total_time += time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
                self.network_total_time += time
            else:
                time = self.time_to_next
                units = time / current_latency
                self.network_total_time += time
                self.next_network_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return (total_delay_units, total_delay_time)

    def do_minimal_download(self, size, min_size, min_time):
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.trace[self.index].bandwidth
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_next * current_bandwidth
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_next -= time
                    self.network_total_time += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                    self.network_total_time += time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    self.network_total_time += time
                    self.next_network_period()
            else:  # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    self.network_total_time += time
                    self.next_network_period()
                else:
                    time = min_time
                    self.time_to_next -= time
                    self.network_total_time += time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delay(self, time):
        while time > self.time_to_next:
            time -= self.time_to_next
            self.network_total_time += self.time_to_next
            self.next_network_period()
        self.time_to_next -= time
        self.network_total_time += time

    def download(self, size, action, first_tile=False, function_check_abandon=None):
        segment = action.segment
        tile = action.tile
        quality = action.quality

        if size <= 0:
            return DownloadProgress(segment=segment, tile=tile, quality=quality,
                                    size=0, downloaded=0,
                                    time=0, time_to_first_bit=0,
                                    abandon=None)
        # 没有check_abandon机制
        if not function_check_abandon or (NetworkModel.min_progress_time <= 0 and
                                          NetworkModel.min_progress_size <= 0):
            latency = 0
            if first_tile:  # 只有first tile才加latency
                latency = self.do_latency_delay(1)
            time = latency + self.do_download(size)
            return DownloadProgress(segment=segment, tile=tile, quality=quality,
                                    size=size, downloaded=size,
                                    time=time, time_to_first_bit=latency,
                                    abandon=None)
        # 有check_abandon机制
        total_download_time = 0
        total_download_size = 0
        min_time_to_progress = NetworkModel.min_progress_time
        min_size_to_progress = NetworkModel.min_progress_size

        if NetworkModel.min_progress_size > 0:
            latency = 0
            if first_tile:
                latency = self.do_latency_delay(1)
            total_download_time += latency
            min_time_to_progress -= total_download_time
            delay_units = 0
        else:
            latency = None
            delay_units = 1

        abandon_action = None
        while total_download_size < size and abandon_action is None:

            if delay_units > 0:
                # NetworkModel.min_progress_size <= 0
                (units, time) = self.do_minimal_latency_delay(delay_units, min_time_to_progress)
                total_download_time += time
                delay_units -= units
                min_time_to_progress -= time
                if delay_units <= 0:
                    latency = total_download_time

            if delay_units <= 0:
                # don't use else to allow fall through
                (bits, time) = self.do_minimal_download(size - total_download_size,
                                                        min_size_to_progress, min_time_to_progress)
                total_download_time += time
                total_download_size += bits
                # no need to upldate min_[time|size]_to_progress - reset below

            dp = DownloadProgress(segment=segment, tile=tile, quality=quality,
                                  size=size, downloaded=total_download_size,
                                  time=total_download_time, time_to_first_bit=latency,
                                  abandon=None)
            if total_download_size < size:
                abandon_action = function_check_abandon(dp)
                min_time_to_progress = NetworkModel.min_progress_time
                min_size_to_progress = NetworkModel.min_progress_size

        return DownloadProgress(segment=segment, tile=tile, quality=quality,
                                size=size, downloaded=total_download_size,
                                time=total_download_time, time_to_first_bit=latency,
                                abandon=abandon_action)


class ThroughputEstimator:
    def __init__(self, config):
        pass

    def push(self, progress):
        raise NotImplementedError


class Ewma(ThroughputEstimator):
    # for throughput:
    default_half_life = [8000, 3000]

    def __init__(self, config):

        super().__init__(config)

        self.throughput = None
        self.latency = None

        if 'ewma_half_life' in config and config['ewma_half_life'] is not None:
            self.half_life = [h * 1000 for h in config['ewma_half_life']]
        else:
            assert (False)
            self.half_life = Ewma.default_half_life

        # TODO: better?
        self.latency_half_life = [h / min(self.half_life) for h in self.half_life]

        self.throughputs = [0] * len(self.half_life)
        self.weight_throughput = 0
        self.latencies = [0] * len(self.latency_half_life)
        self.weight_latency = 0

    def push(self, progress):

        if progress.time <= progress.time_to_first_bit:
            return

        time = progress.time
        tput = progress.downloaded / (progress.time - progress.time_to_first_bit)
        lat = progress.time_to_first_bit

        for i in range(len(self.half_life)):
            alpha = math.pow(0.5, time / self.half_life[i])
            self.throughputs[i] = alpha * self.throughputs[i] + (1 - alpha) * tput

        for i in range(len(self.latency_half_life)):
            alpha = math.pow(0.5, 1 / self.latency_half_life[i])
            self.latencies[i] = alpha * self.latencies[i] + (1 - alpha) * lat

        self.weight_throughput += time
        self.weight_latency += 1

        tput = None
        lat = None
        for i in range(len(self.half_life)):
            zero_factor = 1 - math.pow(0.5, self.weight_throughput / self.half_life[i])
            t = self.throughputs[i] / zero_factor
            tput = t if tput is None else min(tput, t)  # conservative case is min
            zero_factor = 1 - math.pow(0.5, self.weight_latency / self.latency_half_life[i])
            l = self.latencies[i] / zero_factor
            lat = l if lat is None else max(lat, l)  # conservative case is max
        self.throughput = tput
        self.latency = lat

    def get_throughput(self):
        return self.throughput

    def get_latency(self):
        return self.latency


class Session:

    def __init__(self, config):
        self.config = config

        raw_manifest = load_json(config['manifest'])
        self.manifest = ManifestInfo(segment_duration=raw_manifest['segment_duration_ms'],
                                     tiles=raw_manifest['tiles'],
                                     bitrates=raw_manifest['bitrates_kbps'],
                                     utilities=raw_manifest['bitrates_kbps'],
                                     segments=raw_manifest['segment_sizes_bits'])
        del raw_manifest

        raw_network_trace = load_json(config['bandwidth_trace'])
        # TODO: Use latency from bandwidth trace.
        #       Currently always setting latency to 5ms to avoid wasting most of the time waiting for RTT.
        self.network_trace = [NetworkPeriod(time=p['duration_ms'],
                                            bandwidth=p['bandwidth_kbps'],
                                            latency=p['latency_ms'])  # TODO: latency = p['latency_ms']
                              for p in raw_network_trace]
        del raw_network_trace

        raw_pose_trace = load_json(config['pose_trace'])
        self.pose_trace = [PoseInformation(play_time=p['time_ms'],
                                           pose=p['quaternion'])
                           for p in raw_pose_trace]
        del raw_pose_trace

        self.buffer_size = config['buffer_size'] * 1000

        self.buffer = TiledBuffer(self.manifest.segment_duration, self.manifest.tiles)
        self.session_info = SessionInfo(self.manifest, self.buffer, self.buffer_size)

        self.log_file = LogFile(self.session_info, config['log_file'])
        self.session_info.set_log_file(self.log_file)

        self.session_events = SessionEvents()
        self.session_events.add_play_handler(self.play_event)
        self.session_events.add_stall_handler(self.stall_event)
        self.session_events.add_pose_handler(self.pose_event)

        self.video_model = VideoModel(self.manifest)
        self.user_model = UserModel(self.pose_trace)
        self.network_model = NetworkModel(self.network_trace)
        self.headset_model = HeadsetModel(self.session_info, self.session_events)

        self.estimator = Ewma(config)
        self.session_info.set_throughput_estimator(self.estimator)
        # todo: 替换为准确度更高的predictor
        self.viewport_prediction = TestPrediction(self.session_info, self.session_events)
        self.session_info.set_viewport_predictor(self.viewport_prediction)
        # self.viewport_prediction = NavigationGraphPrediction(self.session_info, self.session_events,
        #                                                      config['navigation_graph'])

        # self.session_info.set_viewport_predictor(self.viewport_prediction)
        self.session_info.set_user_model(self.user_model)  # 导入用户的头部运动轨迹

        self.abr = config['abr'](config, self.session_info, self.session_events)

        self.current_stall_time = 0
        self.did_startup = False

        # =============== 新增变量 ==================
        self.total_download_time = 0
        self.qoe_one_step = 0
        self.total_qoe = 0
        self.prev_pose_trace = {}
        self.last_played_segment = 0

    def play_event(self, time):
        # triggered from HeadsetModel
        if not self.did_startup or self.current_stall_time > 0:
            self.log_file.log_play(self.current_stall_time, not self.did_startup)
            self.current_stall_time = 0
            self.did_startup = True

        while time > 0:
            if self.buffer.get_played_segment_partial() == 0 and self.buffer.get_played_segments() < len(
                    self.manifest.segments):
                segment = self.buffer.get_played_segments()
                tiles = [self.buffer.get_buffer_element(segment, t) for t in range(self.manifest.tiles)]
                self.log_file.log_rendering(segment, tiles)  # 日志输出tile质量等级
            do_time = min(time, self.manifest.segment_duration - self.buffer.get_played_segment_partial())
            self.buffer.play_out_buffer(do_time)  # 消耗buffer
            self.session_info.advance_wall_time(do_time)
            time -= do_time

    def stall_event(self, time):
        # triggered from HeadsetModel
        if self.did_startup and self.current_stall_time == 0 and time > 0:
            self.log_file.log_stall()
        self.current_stall_time += time
        self.session_info.advance_wall_time(time)

    def consume_download_time(self, time, time_is_play_time=False):
        # shares self.consumed_download_time with self.run()
        do_time = time - self.consumed_download_time
        assert (do_time >= 0)
        wall_time, stall = self.headset_model.play_for_time(do_time, self.buffer, time_is_play_time=time_is_play_time)
        self.consumed_download_time = time
        return wall_time, stall

    def pose_event(self, value1):
        # triggered from HeadsetModel
        (play_time, pose) = value1
        self.log_file.log_pose(pose)

    def check_abandon(self, download_progress):
        # called during self.network_model.download() call inside self.run()
        wall_time, stall = self.consume_download_time(download_progress.time)
        self.log_file.log_check_abandon(download_progress)
        action = self.abr.check_abandon(download_progress)
        self.log_file.log_check_abandon_verdict(action)
        return action

    def run(self):
        global g_debug_cycle

        video_time = self.manifest.segment_duration * len(self.manifest.segments)
        abandon_action = None
        cycle_index = -1
        tile_prediction_string = None

        # 模拟播放过程
        while self.session_info.buffer.get_play_head() < video_time:
            cycle_index += 1  # 日志文件循环计数
            g_debug_cycle = cycle_index
            self.log_file.log_new_cycle(cycle_index)

            # ----------> 不放弃下载 <----------
            if abandon_action is None:
                tput = self.estimator.get_throughput()  # 带宽估计
                if tput is None:  # initial
                    # TODO: better initial estimate? 是否有更好的初始估计？
                    tput = 0
                self.log_file.log_tput(tput)  # 日志输出-带宽估计

                # 获取最长的播放距离
                view_log_begin = self.buffer.get_played_segments()
                view_log_end = view_log_begin + self.buffer.get_buffer_depth() + 1
                view_log_end = min(view_log_end, len(self.manifest.segments))

                '''=================> 做abr决策 <================='''
                action = self.abr.get_action()  # 返回action列表，包含同一segment下多个tile
                if action is None:
                    old_quality = None
                    self.log_file.log_abr(action)
                else:
                    for i in range(len(action)):
                        old_quality = self.buffer.get_buffer_element(action[i].segment, action[i].tile)
                        if old_quality is None:
                            self.log_file.log_abr(action[i])
                        else:
                            self.log_file.log_abr_replace(action[i], old_quality)
            else:  # already have action lined up from abandon
                # todo: abandon_action时的action是否也为数组？
                action = abandon_action
                for i in range(len(action)):
                    old_quality = self.buffer.get_buffer_element(action[i].segment, action[i].tile)
                    if old_quality is None:
                        self.log_file.log_abr(action[i], from_abandoned=True)
                    else:
                        self.log_file.log_abr(action[i], old_quality, from_abandoned=True)
                abandon_action = None

            delay = 0
            if action is None:
                # pause until the end of the current segment
                delay = self.manifest.segment_duration - self.session_info.buffer.get_played_segment_partial()
                # delay = video_time - self.session_info.buffer.get_play_head()
            elif action[0].delay is not None and action[0].delay > 0:
                delay = action[0].delay

            # 最大buffer不设限
            # if action is not None:
            #     buffer_end = (action[0].segment + 1) * self.manifest.segment_duration
            #     if delay < buffer_end - self.buffer.get_play_head() - self.buffer_size - 0.001:
            #         # 避免缓冲区上溢
            #         delay = buffer_end - self.buffer.get_play_head() - self.buffer_size
            #         self.log_file.log_str('Buffer full: update delay to %.3fs' % (delay / 1000))

            if delay > 0:
                # shares self.consumed_download_time with self.consume_download_time()
                self.consumed_download_time = 0
                self.network_model.delay(delay)
                wall_time, stall_time = self.consume_download_time(delay, time_is_play_time=True)

                ''' 记录每一个segment的pose trace'''
                cur_played_segment = self.buffer.get_played_segments()  # 正在播放的segment
                if self.buffer.get_played_segment_partial() == 0:  # 尚未开始播放
                    cur_played_segment -= 1
                while cur_played_segment >= 0 and self.last_played_segment <= cur_played_segment:
                    start_time = self.last_played_segment * self.manifest.segment_duration
                    end_time = (self.last_played_segment + 1) * self.manifest.segment_duration
                    pose_list = self.user_model.get_pose_in_qoe(start_time, end_time)
                    self.prev_pose_trace[self.last_played_segment] = pose_list[0]

                    self.last_played_segment += 1
                assert self.last_played_segment > cur_played_segment

                self.session_events.trigger_network_delay_event(delay)
                self.log_file.log_delay(delay)

            if action is None:
                continue
            # todo：改成一次性可下载多个tile - finished
            self.total_download_time = 0
            progress_list = []
            bandwidth_usage = 0
            for i in range(len(action)):
                if i == 0:
                    is_first_tile = True
                else:
                    is_first_tile = False
                size = self.manifest.segments[action[i].segment][action[i].tile][action[i].quality]  # 读取待下载的tile的size
                bandwidth_usage += size
                self.consumed_download_time = 0
                ''' 模拟下载过程 '''
                progress = self.network_model.download(size, action[i], is_first_tile,
                                                       self.check_abandon)
                progress_list.append(copy.deepcopy(progress))
                self.total_download_time += progress.time

            ''' 模拟视频播放进程 '''
            wall_time, stall_time = self.consume_download_time(self.total_download_time)

            ''' 记录每一个segment的pose trace'''
            cur_played_segment = self.buffer.get_played_segments()  # 正在播放的segment
            if self.buffer.get_played_segment_partial() == 0:  # 尚未开始播放
                cur_played_segment -= 1
            while cur_played_segment >= 0 and self.last_played_segment <= cur_played_segment:
                start_time = self.last_played_segment * self.manifest.segment_duration
                end_time = (self.last_played_segment + 1) * self.manifest.segment_duration
                pose_list = self.user_model.get_pose_in_qoe(start_time, end_time)
                self.prev_pose_trace[self.last_played_segment] = pose_list[0]

                self.last_played_segment += 1
            assert self.last_played_segment > cur_played_segment

            ''' QoE计算 '''
            segment_idx = action[0].segment
            download_tile = {}
            for i in range(len(action)):
                download_tile[action[i].tile] = action[i].quality

            # 1. 分项指标计算
            start_time = segment_idx * self.manifest.segment_duration
            end_time = (segment_idx + 1) * self.manifest.segment_duration
            pose_list = self.user_model.get_pose_in_qoe(start_time, end_time)
            assert len(pose_list) > 0
            # -----------------------------------------------------
            # quality是视窗内每个tile的平均码率
            bandwidth_usage /= 1024.
            if self.buffer.get_play_head() <= segment_idx * self.manifest.segment_duration:
                delta_quality = self.calculate_delta_quality(pose_list, segment_idx, download_tile)
                delta_var_space = self.calculate_delta_var_space(pose_list, segment_idx, download_tile)
                delta_var_time = self.calculate_delta_var_time(segment_idx)
                bandwidth_wastage = self.calculate_bandwidth_wastage(pose_list, segment_idx, download_tile)
            else:  # 播放后才下载的质量不计入qoe
                delta_quality = 0
                delta_var_space = 0
                delta_var_time = 0
                bandwidth_wastage = bandwidth_usage

            # 2. 线性组合
            # self.qoe_one_step = delta_quality / 8 - 1.85 * stall_time - 0.5 * delta_var_space - 1 * delta_var_time - 0.5 * bandwidth_usage / 8
            self.qoe_one_step = delta_quality / 8 - 1.85 * stall_time - 0.5 * delta_var_space - 1 * delta_var_time - 0.5 * bandwidth_wastage / 8
            self.total_qoe += self.qoe_one_step

            for i in range(len(action)):
                progress = progress_list[i]
                if progress.abandon is None:  # 没有放弃下载
                    self.estimator.push(progress)
                    self.abr.report_action_complete(progress)
                    self.buffer.put_in_buffer(progress.segment, progress.tile, progress.quality)  # 将下载的chunk放入buffer
                    self.log_file.log_download(progress)
                else:  # 放弃下载
                    abandon_action = []
                    self.abr.report_action_cancelled(progress)
                    abandon_action.append(progress.abandon)
                    self.log_file.log_abandoned(progress)
                self.log_file.log_buffer(self.buffer)

    def calculate_delta_quality(self, pose_list, segment_idx, download_tile):
        bitrate = self.manifest.bitrates
        buffer_contents = self.buffer.get_buffer_contents(segment_idx)
        sum_delta_quality = 0
        for i in range(len(pose_list)):
            video_x, video_y = Pose2VideoXY(pose_list[i])
            delta_quality_per_pose = 0
            for tile_idx in download_tile:
                proportion = calculate_viewing_proportion(video_x, video_y, tile_idx)
                if buffer_contents is None or buffer_contents[tile_idx] is None:  # 从未下载过
                    delta_quality_per_pose += proportion * bitrate[download_tile[tile_idx]]
                else:  # 下载过
                    delta_quality_per_pose += proportion * max(
                        bitrate[download_tile[tile_idx]] - buffer_contents[tile_idx], 0)
            sum_delta_quality += delta_quality_per_pose
        delta_quality = sum_delta_quality / len(pose_list)
        self.buffer.add_segment_quality(segment_idx, delta_quality)

        return delta_quality

    def calculate_delta_var_space(self, pose_list, segment_idx, download_tile):
        bitrate = self.manifest.bitrates
        buffer_contents_raw = self.buffer.get_buffer_contents(segment_idx)

        if buffer_contents_raw is None:
            buffer_contents = [None for i in range(TILES_X * TILES_Y)]
            for tile_idx in download_tile:
                buffer_contents[tile_idx] = download_tile[tile_idx]
        else:
            buffer_contents = buffer_contents_raw.copy()
            for tile_idx in download_tile:
                if buffer_contents[tile_idx] is None:
                    buffer_contents[tile_idx] = buffer_contents[tile_idx]
                else:
                    buffer_contents[tile_idx] = max(buffer_contents[tile_idx], download_tile[tile_idx])

        sum_var_space = 0

        # 计算avg
        sum_of_proportion = 0
        for i in range(len(pose_list)):
            video_x, video_y = Pose2VideoXY(pose_list[i])
            for tile_idx in range(len(buffer_contents)):
                if buffer_contents[tile_idx] is not None:  # 遍历下载的tile
                    proportion = calculate_viewing_proportion(video_x, video_y, tile_idx)
                    sum_of_proportion += proportion
        avg_proportion = sum_of_proportion / len(pose_list)
        avg_quality = 0
        if avg_proportion != 0:
            avg_quality = self.buffer.get_segment_quality(segment_idx) / avg_proportion

        for i in range(len(pose_list)):
            video_x, video_y = Pose2VideoXY(pose_list[i])
            var_space_per_pose = 0
            sum_of_proportion = 0
            for tile_idx in range(len(buffer_contents)):
                if buffer_contents[tile_idx] is not None:  # 下载的tile
                    proportion = calculate_viewing_proportion(video_x, video_y, tile_idx)
                    sum_of_proportion += proportion
                    # 空间平滑度，标准差
                    var_space_per_pose += proportion * (bitrate[buffer_contents[tile_idx]] - avg_quality) ** 2
            sum_var_space = 0
            if sum_of_proportion != 0:
                sum_var_space += var_space_per_pose / sum_of_proportion
        quality_var_space = math.sqrt(sum_var_space / len(pose_list))
        last_quality_var_space = self.buffer.get_quality_var_space(segment_idx)
        self.buffer.save_quality_var_space(segment_idx, quality_var_space)
        delta_var_space = quality_var_space - last_quality_var_space  # 计算时需要加上上次的，减掉下次的
        return delta_var_space

    def calculate_delta_var_time(self, segment_idx):
        if segment_idx == 0:
            delta_var_time = 0
        else:
            quality_var_time = 0
            next_quality_var_time = 0
            # todo: update会影响前和后一个segment的质量差，分为前后 [和前面的差，和后面的差]
            prev_quality = self.buffer.get_segment_quality(segment_idx - 1)
            cur_quality = self.buffer.get_segment_quality(segment_idx)
            next_quality = self.buffer.get_segment_quality(segment_idx + 1)
            if prev_quality is None:
                quality_var_time = 0
            else:
                quality_var_time = abs(prev_quality - cur_quality)  # 前后segment平滑度相减

            last_quality_var_time = self.buffer.get_quality_var_time(segment_idx)
            self.buffer.save_quality_var_time(segment_idx, quality_var_time)

            delta_var_time = quality_var_time - last_quality_var_time  # 计算时需要加上上次的，减掉下次的
        return delta_var_time

    def calculate_bandwidth_wastage(self, pose_list, segment_idx, download_tile):
        wastage = 0
        size_info = self.manifest.segments[segment_idx]
        buffer_contents = self.buffer.get_buffer_contents(segment_idx)
        for i in range(len(pose_list)):
            video_x, video_y = Pose2VideoXY(pose_list[i])
            for tile_idx in download_tile:
                proportion = calculate_viewing_proportion(video_x, video_y, tile_idx)
                if proportion == 0:
                    wastage += size_info[tile_idx][download_tile[tile_idx]]
                elif buffer_contents is not None and buffer_contents[tile_idx] is not None:
                    wastage += size_info[tile_idx][buffer_contents[tile_idx]]
        wastage = wastage / (len(pose_list) * 1024.)
        return wastage

    def get_total_qoe(self):
        return self.total_qoe


if __name__ == '__main__':
    # TODO: parse arguments for config

    default_config = {}
    default_config['ewma_half_life'] = [4, 1]  # seconds
    default_config['buffer_size'] = 5  # seconds
    default_config['manifest'] = 'video/manifest/movie360.json'
    default_config['navigation_graph'] = 'cu_navigation_graph.json'
    default_config['bandwidth_trace'] = 'network/network.json'
    default_config['pose_trace'] = 'video/pose_trace/pose_trace.json'
    default_config['log_file'] = 'log/session.log'

    default_config['bola_know_per_segment_sizes'] = True
    default_config['bola_use_placeholder'] = True
    default_config['bola_allow_replacement'] = True
    default_config['bola_insufficient_buffer_safety_factor'] = 0.5
    default_config['bola_minimum_tile_weight'] = 0.5 / 16

    # default_config['abr'] = BaselineAbr
    default_config['abr'] = TestAbr

    config = default_config.copy()

    float_args = ['buffer_size', 'bola_insufficient_buffer_safety_factor', 'bola_minimum_tile_weight']
    list_float_args = ['ewma_half_life']
    bool_args = ['bola_know_per_segment_sizes', 'bola_use_placeholder', 'bola_allow_replacement']

    for arg in sys.argv[1:]:
        entry = arg.split('=')
        bad_argument = False
        try:
            if len(entry) != 2 or entry[0] not in config:
                bad_argument = True
            elif entry[0] in float_args:
                config[entry[0]] = float(entry[1])
            elif entry[0] in list_float_args:
                if entry[1][0] != '[' or entry[1][-1] != ']':
                    bad_argument = True
                else:
                    config[entry[0]] = [float(f) for f in entry[1][1].split(',')]
            elif entry[0] in bool_args:
                if entry[1].lower() in ['t', 'true', 'y', 'yes']:
                    config[entry[0]] = True
                elif entry[1].lower() in ['f', 'false', 'n', 'no']:
                    config[entry[0]] = False
                else:
                    bad_argument = True
            elif entry[0] == 'abr':
                if entry[1] == 'TrivialThroughputAbr':
                    config[entry[0]] = TrivialThroughputAbr
                elif entry[1] == 'BaselineAbr':
                    config[entry[0]] = BaselineAbr
                elif entry[1] == 'ThreeSixtyAbr':
                    config[entry[0]] = ThreeSixtyAbr
                else:
                    bad_argument = True
            else:
                config[entry[0]] = entry[1]
        except:
            bad_argument = True
            raise

        if bad_argument:
            print('Bad argument: "%s"' % arg, file=sys.stderr)

    session = Session(config)
    session.run()
    total_qoe = session.total_qoe
    print("total qoe = %.2f" % total_qoe)
