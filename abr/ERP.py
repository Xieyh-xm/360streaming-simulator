''' ###########################
    #  低质量全画幅（算法下限）  #
    ###########################'''

from collections import namedtuple
from abr.TiledAbr import TiledAbr
from sabre360_with_qoe import SessionInfo, SessionEvents

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')
TILES_X = 8
TILES_Y = 8

SLEEP_PERIOD = 500  # 暂停时长 ms


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


class ERP(TiledAbr):
    def __init__(self, config, session_info: SessionInfo, session_events: SessionEvents):
        self.session_info = session_info
        self.session_events = session_events

        self.manifest = self.session_info.get_manifest()
        self.buffer = self.session_info.get_buffer()
        self.latest_seg = -1
        self.video_size = self.manifest.segments

    def get_action(self):
        ''' 下载低质量全画幅的内容 '''
        max_seg_idx = len(self.video_size)-1
        self.latest_seg += 1
        self.latest_seg = min(self.latest_seg, max_seg_idx + 1)
        if self.latest_seg > max_seg_idx:
            action = [TiledAction(0, 0, 0, SLEEP_PERIOD)]
            return action
        action = []
        for tile_id in range(TILES_X * TILES_Y):
            action.append(TiledAction(self.latest_seg, tile_id, 0, 0))
        return action
