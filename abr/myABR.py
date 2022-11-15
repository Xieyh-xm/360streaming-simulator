import math
from collections import namedtuple
from abr.TiledAbr import TiledAbr

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action

# todo: abr for testing environment
class TestAbr(TiledAbr):
    '''
    用于测试的ABR算法，没有视角预测模块，以最低码率下载下一个segment的所有tile
    '''

    def __init__(self, config, session_info, session_events):
        # print("<< test abr >>")
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)

    def get_action(self):
        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        buffer = self.session_info.get_buffer()
        segment0 = buffer.get_played_segments()
        depth = buffer.get_buffer_depth()  # how many segments in buffer (one tile in segment enough to count)
        begin = 0
        if buffer.get_played_segment_partial() > 0:
            begin = 1
        end = depth + 1
        end = min(end, self.max_depth + 1)  # allow max_depth + 1, but that action requires delay
        end = min(end, segment_count - segment0)
        begin += segment0
        end += segment0

        segment = None
        tile = None
        blank_tiles = set()

        ''' 1. 确定segment '''
        # select the next segment where at least one tile is None
        for s in range(begin, end):
            bits_used = 0
            for t in range(manifest.tiles):  # 遍历每个tile
                if buffer.get_buffer_element(s, t) is None:  # 尚未被下载的tile
                    segment = s
                    blank_tiles.add(t)
                else:  # 该segment已下载的总码率
                    bits_used += manifest.segments[s][t][buffer.get_buffer_element(s, t)]
            # if there is something in blank tiles we have found our segment
            if len(blank_tiles) > 0:
                break

        # if segment is None, return None 没有可下载的segment
        if segment is None:
            action = []
            action.append(TiledAction(-1, -1, -1, 500))
            return action

        ''' 2. 确定码率 '''
        qualities = self.allocate_quality(0, segment)
        ''' 3. 确定tile '''
        for t in blank_tiles:
            tile = t
        # ABR输出 — segment | 待下载的tile | 对应的码率 | delay
        action = []
        for tile in blank_tiles:
            action.append(TiledAction(segment, tile, qualities[tile], 0))
        # action.append(TiledAction(segment, tile, qualities[tile], 0))
        if len(action) == 0:
            action.append(TiledAction(-1, -1, -1, 500))
        return action

    def allocate_quality(self, bits, segment):
        ''' 给画面内的所有tile分配最低码率 '''
        manifest = self.session_info.get_manifest()
        buffer = self.session_info.get_buffer()
        sizes = manifest.segments[segment]
        qualities = [bits] * manifest.tiles

        return qualities
