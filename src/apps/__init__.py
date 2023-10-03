from .app import App
from .classification_app import ClassificationApp
from .fedl2p_classification_app import FedL2PClassificationApp

from src.utils import get_func_from_config

import logging
logger = logging.getLogger(__name__)

def get_app(ckp):
    app_config = ckp.config.app
    app_fn = get_func_from_config(app_config)

    return app_fn(ckp, **app_config.args)
