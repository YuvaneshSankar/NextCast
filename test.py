import random
from datetime import datetime, timedelta
from logger import init_log, log_result

from InventoryDemandClassifier import InventoryDemandClassifier

def random_timestamp():
    base = datetime(2024, 1, 1)
    random_days = random.randint(0, 200)
    return (base + timedelta(days=random_days)).isoformat()


classifier = InventoryDemandClassifier()
init_log()