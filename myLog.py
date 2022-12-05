import logging


class myLog:
    def __init__(self, path):
        # 创建一个logger
        self.path = path
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # 创建handle，用于写入日志文件
        self.logfile = path
        self.fh = logging.FileHandler(self.logfile, mode='w')
        self.fh.setLevel(logging.DEBUG)

        # 创建handle，用于输出到控制台
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)

        # 定义handle的输出格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

    def log_ppo_output(self, output):
        self.logger.info("ppo output : {}".format(output))

    def log_sleep(self, sleep_time):
        self.logger.info("ACTION : Sleep {} ms\n".format(sleep_time))

    def log_bt_action(self, segment):
        self.logger.info("ACTION : BT Download ( Segment = {} )\n".format(segment))

    def log_et_action(self, segment, level):
        self.logger.info("ACTION : ET Download ( Segment = {} \tLevel = {})\n".format(segment, level))

    def log_bw_mask(self, mask):
        self.logger.info("bw mask : {}".format(mask))

    def log_playhead(self, playhead):
        self.logger.info("playhead : {}".format(playhead / 1000.))

    def log_first_et(self, segment):
        self.logger.info("first et seg : {}".format(segment))

    def log_state(self, state):
        state = state[0]
        # 吞吐量
        self.logger.info("tput : {}".format(state[0:10]))
        self.logger.info("bt buffer leng: {} \t et buffer len : {}".format(state[10], state[11]))
        self.logger.info("BT bitrate : {}".format(state[12]))
        self.logger.info("ET bitrate : {}".format(state[13:18]))
        self.logger.info("tile num : {}".format(state[18:23]))
        self.logger.info("avg level : {}".format(state[24:29]))
        self.logger.info("fov speed at X : {} \t fov speed at Y : {}".format(state[29], state[30]))
        self.logger.info("pred acc : {}".format(state[31:36]))

    def log_stall(self, stall_time):
        self.logger.info("stall time : {} ms".format(stall_time))

    def log_metric(self, metrics):
        self.logger.info("------------------------------------------------------------------")
        self.logger.info('Score: {:.2f}'.format(metrics[0]))
        self.logger.info('QoE: {:.2f}\t\tbandwidth_usage: {:.2f}'.format(metrics[1], metrics[6]))
        self.logger.info('Quality: {:.2f}\tStall time: {:.2f}\t'.format(metrics[2], metrics[3]))
        self.logger.info('Oscillation in space: {:.2f}\tOscillation in time: {:.2f}'.format(metrics[4], metrics[5]))
        wastage_ratio = metrics[7] / metrics[6]
        self.logger.info('Bandwidth wastage: {:.2f}'.format(metrics[7]))
        self.logger.info('Wastage ratio: {:.2f}\n'.format(wastage_ratio))

    def log_download_time(self, download_time):
        self.logger.info('download time : {}'.format(download_time))


if __name__ == '__main__':
    test_log = myLog('log/test')
    # test_log.log_sleep()
