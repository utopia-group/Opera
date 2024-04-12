from b2s.stats import Stats


class Converter:

    def convert(self, src: str) -> Stats:
        raise NotImplementedError
    
    def __init__(self, **_) -> None:
        pass