from dataclasses import asdict, dataclass
import json
import os


@dataclass
class BaseState:
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def create_default(cls, path: str):
        cls.save(cls(), path)
        return cls()
    
    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            return cls.create_default(path)

        with open(path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                setattr(cls, key, value)
        return cls(**data)
