from timeit import default_timer as timer


class Timer:
    def __init__(self):
        self._timers = {}
        self._times = {}
        self._current_checkpoint = None

    def checkpoint(self, checkpoint_name):
        if self._current_checkpoint is not None:
            self._end_checkpoint(self._current_checkpoint)
        self._start_checkpoint(checkpoint_name)

    def stop(self):
        if self._current_checkpoint is not None:
            self._end_checkpoint(self._current_checkpoint)

    def times(self, fracs=False):
        if fracs:
            total_time = sum(self._times.values())
            return {k: v / total_time for k, v in self._times.items()}
        else:
            return dict(self._times)

    def __del__(self):
        self.stop()

    def _start_checkpoint(self, checkpoint_name):
        self._current_checkpoint = checkpoint_name
        self._timers[checkpoint_name] = timer()

    def _end_checkpoint(self, checkpoint_name):
        start_timer = self._timers[checkpoint_name]
        del self._timers[checkpoint_name]
        end_timer = timer()

        if checkpoint_name in self._times:
            self._times[checkpoint_name] += end_timer - start_timer
        else:
            self._times[checkpoint_name] = end_timer - start_timer

        self._current_checkpoint = None
