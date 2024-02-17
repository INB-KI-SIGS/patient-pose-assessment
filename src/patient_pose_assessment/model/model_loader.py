import logging
from copy import deepcopy
from logging import Handler, LogRecord
from typing import Dict, Optional

import yaloader
from mllooper import State, Module, ModuleConfig
from mllooper.logging.messages import ModelLogMessage
from mllooper.models import Model


class ModelLogHandler(Handler):
    def __init__(self):
        super().__init__()
        self.model_state_dicts: Dict[str, Dict] = {}

    def emit(self, record: LogRecord) -> None:
        # Skip if it isn't a subclass of `ModelLogMessage`
        if isinstance(record.msg, ModelLogMessage):
            model_log: ModelLogMessage = record.msg

            model_state_dict = deepcopy(model_log.model.state_dict())
            self.model_state_dicts[model_log.name] = model_state_dict


class ModelLoader(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handler = ModelLogHandler()
        self.handler.setLevel(0)
        logging.getLogger().addHandler(self.handler)

        self.model: Optional[Model] = None

    def initialise(self, modules: Dict[str, "Module"]) -> None:
        try:
            model = modules["Evaluator"].modules["model"]
            assert isinstance(model, Model)
            self.model = model
        except KeyError:
            raise KeyError(f"{self.name} needs a model to be in the initialization dictionary.")

    def step(self, state: State) -> None:
        if self.model.name in self.handler.model_state_dicts:
            model_state_dict = self.handler.model_state_dicts[self.model.name]
            self.model.module.load_state_dict(model_state_dict)
            del self.handler.model_state_dicts[self.model.name]
            self.logger.info(f"Loaded model for {self.model.name}")


@yaloader.loads(ModelLoader)
class ModelLoaderConfig(ModuleConfig):
    name: str = "ModelLoader"
