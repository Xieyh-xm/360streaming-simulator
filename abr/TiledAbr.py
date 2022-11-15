class TiledAbr:

    # TODO: rewrite report_*() to use SessionEvents

    def __init__(self):
        pass

    def get_action(self):
        raise NotImplementedError

    def check_abandon(self, progress):
        return None

    def report_action_complete(self, progress):
        pass

    def report_action_cancelled(self, progress):
        pass

    def report_seek(self, where):
        raise NotImplementedError
