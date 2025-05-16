import time

class Profile:
    _header = '------------------------------------------------------------'
    # _time
    
    def __init__(self, label):
        print(self._header)
        print(f'{repr(label):20}:', end=' ')        
        
    def __enter__(self):
        self._time = time.time()
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None:
            print(f'{time.time() - self._time:.5f}s')
        else:
            print(f'Failed due to: {exc_type.__name__}')
        print(self._header)
