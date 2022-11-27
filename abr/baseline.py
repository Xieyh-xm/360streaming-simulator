import math
from collections import namedtuple

# from sabre360 import debug_log_level
debug_log_level = True
# delay happens before download
TiledAction = namedtuple('TiledAction', 'segment tile quality delay')


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


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


# TODO: Avoid redundant work
"""
A baseline algorithm where bits are allocated from low to high quality in the order of highest to lowest prediction
probability.

"""


class BaselineAbr(TiledAbr):

    def __init__(self, config, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)

    def get_action(self):

        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        bitrate_count = len(manifest.bitrates)
        buffer = self.session_info.get_buffer()
        segment0 = buffer.get_played_segments()
        sizes = self.session_info.get_manifest().segments[segment0]
        depth = buffer.get_buffer_depth()  # how many segments in buffer (one tile in segment enough to count)
        begin = 0
        if buffer.get_played_segment_partial() > 0:
            begin = 1
        end = depth + 1
        end = min(end, self.max_depth + 1)  # allow max_depth + 1, but that action requires delay
        end = min(end, segment_count - segment0)
        begin += segment0
        end += segment0

        quality = 0
        tput = self.session_info.get_throughput_estimator().get_throughput()
        action = None
        segment = None
        tile = None
        blank_tiles = set()

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
            return None
        #
        # if tput is None:
        #     return TiledAction(segment, None, None, 0)
        ''' 视角预测模块 '''
        tile_probabilities = self.session_info.get_viewport_predictor().predict_tiles(segment)
        if tput is not None:
            qualities = self.allocate_quality(tput * self.session_info.get_manifest().segment_duration - bits_used,
                                              segment, tile_probabilities, blank_tiles)
        else:
            qualities = self.allocate_quality(0, segment,
                                              tile_probabilities, blank_tiles)
        for t in blank_tiles:
            if tile is None or tile_probabilities[t] > tile_probabilities[tile]:
                tile = t
        action = []
        action.append(TiledAction(segment, tile, qualities[tile], 0))
        return action

    # what if there is not enough bits to give lowest quality for all tiles?
    def allocate_quality(self, bits, segment, tile_probabilities, blank_tiles):

        manifest = self.session_info.get_manifest()
        buffer = self.session_info.get_buffer()
        sizes = manifest.segments[segment]
        qualities = [0] * manifest.tiles
        curr_tile = 0
        remaining_bits = bits

        for tile in blank_tiles:
            remaining_bits -= sizes[tile][qualities[tile]]
        while remaining_bits > 0:
            found_one = False
            for tile in blank_tiles:
                # find the tile that meets this criteria
                if qualities[tile] < len(manifest.bitrates) - 1:
                    difference = sizes[tile][qualities[tile] + 1] - sizes[tile][qualities[tile]]
                if qualities[tile] < len(manifest.bitrates) - 1 and \
                        tile_probabilities[tile] > 0.0 and \
                        sizes[tile][qualities[tile] + 1] - sizes[tile][qualities[tile]] <= remaining_bits and \
                        (not found_one or
                         (tile_probabilities[tile] / sizes[tile][qualities[tile] + 1]) > \
                         (tile_probabilities[curr_tile] / sizes[curr_tile][qualities[curr_tile] + 1])):
                    curr_tile = tile
                    found_one = True
            # if no tile is found we are done
            if not found_one:
                break
            qualities[curr_tile] += 1
            remaining_bits -= sizes[curr_tile][qualities[curr_tile]] - sizes[curr_tile][qualities[curr_tile] - 1]

        return qualities


class TrivialThroughputAbr(TiledAbr):

    def __init__(self, config, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)

    def get_action(self):

        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        bitrate_count = len(manifest.bitrates)
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

        quality = 0
        tput = self.session_info.get_throughput_estimator().get_throughput()
        if tput is not None:
            if depth <= 1:
                safety_factor = 0.5
            elif depth <= 2:
                safety_factor = 0.6
            elif depth <= 3:
                safety_factor = 0.75
            else:
                safety_factor = 0.9
            for bitrate in manifest.bitrates[1:]:
                if safety_factor * tput >= bitrate:
                    quality += 1

        possible_actions = []
        for segment in range(begin, end):
            for tile in range(manifest.tiles):
                old_quality = buffer.get_buffer_element(segment, tile)
                # Note that when segment == segment0 + depth we are exploring expanding the buffer, old_quality is always None in that case
                if old_quality is None:
                    # add new-download actions
                    possible_actions += [(segment, tile, quality, None)
                                         for quality in range(bitrate_count)]
                else:
                    # add replacement actions
                    possible_actions += [(segment, tile, quality, old_quality)
                                         for quality in range(old_quality + 1, bitrate_count)]
        # filter possible actions by chosen quality level
        possible_actions = [action for action in possible_actions if action[2] == quality]
        best_action = None
        best_score = None
        for action in possible_actions:
            action_score = -action[0]  # prefer to download earlier segments
            if action[3] is None:
                action_score += self.max_depth + 2  # give higher priority to new downloads vs replacements

            if best_action is None or action_score > best_score:
                best_action = action
                best_score = action_score

        if best_action is None:
            # happens when we reach end of video
            return None
        else:
            delay = 0
            if best_action[0] == segment0 + self.max_depth:
                delay = manifest.segment_duration - buffer.get_played_segment_partial()
            action = []
            action.append(TiledAction(segment=best_action[0], tile=best_action[1], quality=best_action[2], delay=delay))
            return action

    def check_abandon(self, progress):
        # TODO
        return None

    def report_action_complete(self, progress):
        # TODO
        pass

    def report_action_cancelled(self, progress):
        # TODO
        pass


# TODO: design an algorithm worthy of a new name
class ThreeSixtyAbr(TiledAbr):

    # BOLA-based ABR

    def __init__(self, config, session_info, session_events):
        self.know_per_segment_sizes = config['bola_know_per_segment_sizes']
        self.use_placeholder = config['bola_use_placeholder']
        self.allow_replacement = config['bola_allow_replacement']
        self.insufficient_buffer_safety_factor = config['bola_insufficient_buffer_safety_factor']
        self.minimum_tile_weight = config['bola_minimum_tile_weight']

        self.session_info = session_info
        self.session_events = session_events

        manifest = session_info.get_manifest()

        min_buffer_low = 2 * manifest.segment_duration

        if self.use_placeholder:
            self.buffer_low = session_info.buffer_size - manifest.segment_duration * len(manifest.bitrates)
            self.buffer_high = session_info.buffer_size - manifest.segment_duration
            if self.buffer_low < min_buffer_low:
                self.buffer_high += min_buffer_low - self.buffer_low
                self.buffer_low = min_buffer_low
        else:
            self.buffer_low = session_info.buffer_size - manifest.segment_duration * len(manifest.bitrates)
            self.buffer_high = session_info.buffer_size - manifest.segment_duration
            if self.buffer_low < min_buffer_low:
                self.buffer_low = min(min_buffer_low, self.buffer_high / 2)

        # Note: we can use bitrates instead of bits to calculate Vp and gp: 4g-scaling the size does not affect them
        self.bitrates = manifest.bitrates
        self.utilities = [math.log(bitrate / self.bitrates[0]) for bitrate in self.bitrates]
        self.average_bits = [bitrate * manifest.segment_duration / manifest.tiles for bitrate in self.bitrates]
        alpha = ((self.bitrates[0] * self.utilities[1] - self.bitrates[1] * self.utilities[0]) /
                 (self.bitrates[1] - self.bitrates[0]))
        self.Vp = (self.buffer_high - self.buffer_low) / (alpha + self.utilities[-1])
        self.gp = ((alpha * self.buffer_high + self.utilities[-1] * self.buffer_low) /
                   (self.buffer_high - self.buffer_low))

        session_events.add_network_delay_handler(self.network_delay_event)
        self.placeholder_buffer = 0

        log_file = self.session_info.get_log_file()
        log_file.log_str('BOLA Vp = %.0f' % self.Vp)
        log_file.log_str('BOLA gp = %.3f' % self.gp)
        log_file.log_str('BOLA target buffer: %.3f-%.3f%s' %
                         (self.buffer_low / 1000, self.buffer_high / 1000,
                          ' (may include buffer expansion)' if self.use_placeholder else ''))

    def network_delay_event(self, time):
        if self.use_placeholder:
            self.placeholder_buffer += time

    def rho(self, bits, utility, buffer_level):
        return (self.Vp * (utility + self.gp) - buffer_level) / bits

    def rho_inc(self, bits, utility, bits_inc, utility_inc, buffer_level, scalable=False):
        # Math outline:
        # Consider qualities having (bits, utility) and (bits + bits_inc, utility + utility_inc).
        # There is some buffer level for which:
        #     rho(bits, utility) == rho(bits + bits_inc, utility + utility_inc).
        # That is:
        #     (Vp * (utility + gp) - buffer_level) / bits
        #         == (Vp * (utility + utility_inc + gp) - buffer_level) / (bits + bits_inc).
        # Solving the above equation gives:
        #     buffer_level = Vp * (utility + gp) - Vp * bits * utility_inc / bits_inc.
        # Assume there exists an incremental download,
        # then we want the following expression to match the above two expressions at buffer_level:
        #     (Vp * u - buffer_level) / bits_inc
        #         == (Vp * (utility + gp) - buffer_level) / bits
        #         == (Vp * (utility + utility_inc + gp) - buffer_level) / (bits + bits_inc).
        # Solving the above equations gives:
        #     u == utility + utility_inc - bits * utility_inc / bits_inc + gp.
        # The objective function for such an incremental download would be:
        #     (Vp * (utility + utility_inc - bits * utility_inc / bits_inc + gp) - buffer_level) / bits_inc.
        # However, incremental downloads are only available for scalable downloads.
        # If scalable downloads are not an option, the denominator must reflect a full new download:
        #     (Vp * (utility + utility_inc + gp - bits * utility_inc / bits_inc) - buffer_level)
        #         / (bits + bits_inc).
        if scalable:
            download_bits = bits_inc
        else:
            download_bits = bits + bits_inc
        return (self.Vp * (utility + utility_inc - bits * utility_inc / bits_inc + self.gp) - buffer_level) / \
               download_bits

    def delay_for_positive_rho(self, utility, buffer_level):
        buffer_target = self.Vp * (utility + self.gp)
        if buffer_level > buffer_target:
            return buffer_level - buffer_target
        else:
            return 0

    def get_action(self):
        global g_debug_cycle

        scalable = False  # TODO: update when we start supporting scalable video
        assert (self.use_placeholder or self.placeholder_buffer == 0)
        if debug_log_level and self.use_placeholder:
            self.session_info.get_log_file().log_str('DEBUG BOLA placeholder: %.3f' % (self.placeholder_buffer / 1000))

        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        bitrate_count = len(manifest.bitrates)

        buffer = self.session_info.get_buffer()
        buffer_play_head = buffer.get_play_head()

        # check buffer capacity
        buffer_full_level = buffer_play_head + self.session_info.buffer_size
        buffer_full_segment = math.floor(buffer_full_level / manifest.segment_duration)
        buffer_full_delay = (buffer_full_segment + 1) * manifest.segment_duration - buffer_full_level
        assert (0 < buffer_full_delay <= manifest.segment_duration)

        # first segment to consider is first segment in buffer
        begin = buffer.get_played_segments()
        # if first segment started rendering it is too late to update it
        segment0_partial = buffer.get_played_segment_partial()
        if segment0_partial > 0:
            begin += 1

        # last segment to consider (end is exclusive)
        end = min(buffer_full_segment + 1, segment_count)

        throughput = self.session_info.get_throughput_estimator().get_throughput()
        latency = self.session_info.get_throughput_estimator().get_latency()

        assert ((throughput is None) == (latency is None))
        safety_throughput = 0 if throughput is None else self.insufficient_buffer_safety_factor * throughput
        safety_latency = 0 if latency is None else latency
        (incomplete_segment, incomplete_tiles, safe_bits_available) = \
            self.get_insufficient_buffer_list(safety_throughput, safety_latency)
        # Note that get_insufficient_buffer_list() does its own buffer level calculation.
        # It is important to not use the placeholder buffer in get_insufficient_buffer_list().

        if debug_log_level:
            self.session_info.get_log_file().log_str(
                'DEBUG Safety: safe_throughput=%.0f safe_latency=%d segment=%s tiles=%s bits=%s'
                % (safety_throughput, safety_latency,
                   incomplete_segment if incomplete_segment is not None else '-',
                   str(tuple(incomplete_tiles)) if incomplete_tiles is not None else '()',
                   round(safe_bits_available) if safe_bits_available is not None else '-'))

        view_predictor = self.session_info.get_viewport_predictor()

        best_objective = None
        best_action = None
        best_action_placeholder_delay = 0
        # We can have (best_objective is None and best_action is not
        # None) when the best action has a negative objective and
        # needs a delay to push its objective up to zero.
        for segment in range(begin, end):
            real_buffer_level = segment * manifest.segment_duration - buffer_play_head
            buffer_level = real_buffer_level + self.placeholder_buffer
            verbose_buffer_level = '(real=%d*%.3f-%.3f=%.3f)+(placeholder=%.3f)=%.3fs' % (
                segment, manifest.segment_duration / 1000, buffer_play_head / 1000, real_buffer_level / 1000,
                self.placeholder_buffer / 1000, buffer_level / 1000)
            assert (buffer_level >= 0)

            min_delay = 0
            if segment == buffer_full_segment:
                min_delay = buffer_full_delay

            # Using (self.minimum_tile_weight + prob) instead of
            # max(self.minimum_tile_weight, prob) does not give a
            # result within the range (0, 1), but (a) it does not
            # matter because the values are only used to multiply
            # objective function values to compare decisions against
            # each other. (b) It still ensures that each tile has a
            # non-zero value and (c) also preserves some difference
            # between tiles with very low probability.
            tile_prediction = [self.minimum_tile_weight + prob for prob in view_predictor.predict_tiles(segment)]

            for tile in range(manifest.tiles):
                quality_bits = manifest.segments[segment][tile] if self.know_per_segment_sizes else self.average_bits
                bits0 = quality_bits[0]

                old_quality = buffer.get_buffer_element(segment, tile)
                if old_quality is not None and not self.allow_replacement:
                    continue
                if old_quality is not None:
                    old_bits = quality_bits[old_quality]
                    old_objective = self.rho(old_bits, self.utilities[old_quality], buffer_level)

                # TODO: While we check all tile possibilities, we can optimize by pruning too far in the future
                #       Note that the following commented block might change behavior in some cases because
                #       prob(segment, tile) can be greater than prob(segment - 1, tile)
                ## if old_quality is None and segment > begin and buffer.get_buffer_element(segment - 1, tile) is None:
                ##     # we already considered adding a new download for this tile
                ##     continue

                for quality in range(0 if old_quality is None else old_quality + 1, bitrate_count):

                    bits = quality_bits[quality]

                    # check insufficient buffer rule
                    if safe_bits_available is not None:
                        safe_bits = safe_bits_available
                        if segment == incomplete_segment and tile in incomplete_tiles:
                            assert (old_quality is None)
                            safe_bits += bits0
                        if bits > safe_bits:
                            continue

                    if old_quality is not None and scalable:
                        download_bits = max(0, bits - old_bits)  # do not rely on monotonic bits with quality
                    else:
                        download_bits = bits

                    # Make sure we have enough time to download tile
                    # Exception: do not block tiles in incomplete_tiles
                    if (throughput is not None and download_bits / throughput > real_buffer_level and
                            (segment != incomplete_segment or tile not in incomplete_tiles)):
                        continue

                    objective = self.rho(bits, self.utilities[quality], buffer_level)

                    if old_quality is not None:
                        # replacement download
                        reference_objective = objective
                        objective = self.rho_inc(old_bits, self.utilities[old_quality],
                                                 bits - old_bits, self.utilities[quality] - self.utilities[old_quality],
                                                 buffer_level, scalable=scalable)
                        if objective < 0 or reference_objective < 0 or reference_objective <= old_objective:
                            continue

                    if objective >= 0 and min_delay == 0:
                        objective *= tile_prediction[tile]
                        if best_objective is None or objective > best_objective:
                            best_objective = objective
                            best_action = TiledAction(segment=segment, tile=tile, quality=quality, delay=0)
                            best_action_placeholder_delay = 0
                    else:
                        # buffer too full for positive objective
                        assert (old_quality is None)
                        # if we already had a non-delay action with an objective > 0, don't attemped delay action
                        if best_objective is None:
                            shrink_buffer = self.delay_for_positive_rho(self.utilities[quality], buffer_level)
                            v = (', buffer_level=%s, delay_for_positive_rho(utility=%.3f, buffer_level=%.3f) = %.3f' %
                                 (verbose_buffer_level, self.utilities[quality], buffer_level / 1000,
                                  shrink_buffer / 1000))
                            v += (', shrink_buffer = max(min_delay=%.3fs, delay_for_positive_rho=%.3fs)' %
                                  (min_delay / 1000, shrink_buffer / 1000))
                            shrink_buffer = max(min_delay, shrink_buffer)
                            v += (', placeholder_delay=min(placeholder_buffer=%.3fs, shrink_buffer-min_delay=%.3fs)' %
                                  (self.placeholder_buffer / 1000, (shrink_buffer - min_delay) / 1000))
                            placeholder_delay = min(self.placeholder_buffer, shrink_buffer - min_delay)
                            v += (', delay = (shrink_buffer=%.3fs) - (placeholder_delay=%.3fs))' %
                                  (shrink_buffer / 1000, placeholder_delay / 1000))
                            delay = shrink_buffer - placeholder_delay
                            assert (delay >= min_delay)

                            if (best_action is None or
                                    delay < best_action.delay or
                                    (delay == best_action.delay and placeholder_delay < best_action_placeholder_delay)):
                                # leave best_objective = None
                                best_action = TiledAction(segment=segment, tile=tile,
                                                          quality=quality, delay=delay)
                                best_action_placeholder_delay = placeholder_delay
                                verbose = v

        if debug_log_level and best_action_placeholder_delay > 0:
            self.session_info.get_log_file().log_str('DEBUG BOLA placeholder shrink by %.3fs %s' %
                                                     (best_action_placeholder_delay / 1000, verbose))

        assert (best_action_placeholder_delay == 0 or best_objective is None)
        self.placeholder_buffer -= best_action_placeholder_delay
        return [best_action]

    def get_insufficient_buffer_list(self, safety_throughput, latency):
        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)

        buffer = self.session_info.get_buffer()

        # first segment to consider is first segment in buffer
        begin = buffer.get_played_segments()
        # if first segment started rendering it is too late to update it
        if buffer.get_played_segment_partial() > 0:
            begin += 1

        # last segment to consider (end is exclusive)
        end = segment_count

        incomplete_segment = None
        incomplete_tiles = set()
        for segment in range(begin, end):
            for tile in range(manifest.tiles):
                if buffer.get_buffer_element(segment, tile) is None:
                    incomplete_tiles.add(tile)
            if len(incomplete_tiles) > 0:
                incomplete_segment = segment
                break

        if incomplete_segment is None:
            # we have all tiles till the very end
            return (None, None, None)

        time_to_incomplete = incomplete_segment * manifest.segment_duration - buffer.get_play_head()
        assert (time_to_incomplete >= 0)
        bits_to_incomplete = safety_throughput * (time_to_incomplete - latency * (len(incomplete_tiles) + 1))
        if self.know_per_segment_sizes:
            bits_for_tiles = sum([manifest.segments[incomplete_segment][tile][0] for tile in incomplete_tiles])
        else:
            bits_for_tiles = manifest.bitrates[0] * manifest.segment_duration * len(incomplete_tiles) / manifest.tiles
        bits_available = max(0, bits_to_incomplete - bits_for_tiles)

        return (incomplete_segment, incomplete_tiles, bits_available)

    def check_abandon(self, progress):
        if progress.segment <= self.session_info.get_buffer().get_played_segments():
            return self.get_action()
        # TODO
        return None

    def report_action_complete(self, progress):
        # TODO
        pass

    def report_action_cancelled(self, progress):
        # TODO
        pass
