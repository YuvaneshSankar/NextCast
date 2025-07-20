
class InventoryDemandClassifier:
    def __init__(self, alpha=0.95):
        self.alpha = alpha

        self.avg_demand_pct = 0.3
        self.avg_inventory_pct = 0.7
        self.avg_risk_ratio = 0.4  # demand / current

    def update_values(self, old_val, new_val):
        return self.alpha * old_val + (1 - self.alpha) * new_val
    
    def classify(self, timestamp,total, current, demand):
        if total == 0 or current == 0:
            return "Invalid"

        demand_pct = demand / total
        inventory_pct = current / total
        risk_ratio = demand / current

        self.avg_demand_pct = self.update_values(self.avg_demand_pct, demand_pct)
        self.avg_inventory_pct = self.update_values(self.avg_inventory_pct, inventory_pct)
        self.avg_risk_ratio = self.update_values(self.avg_risk_ratio, risk_ratio)

        critical_demand_pct = self.avg_demand_pct + 0.2
        critical_inventory_pct = self.avg_inventory_pct - 0.3

        warning_demand_pct = self.avg_demand_pct
        warning_inventory_pct = self.avg_inventory_pct

        if (
            demand > current or 
            inventory_pct < critical_inventory_pct or 
            demand_pct > critical_demand_pct
        ):
            status = "Critical"
        elif (
            demand_pct >= warning_demand_pct or 
            inventory_pct < warning_inventory_pct
        ):
            status = "Warning"
        else:
            status = "Normal"

        return {
            "timestamp": timestamp,
            "status": status,
            "demand_pct": round(demand_pct, 3),
            "inventory_pct": round(inventory_pct, 3),
            "risk_ratio": round(risk_ratio, 3),
            "avg_demand_pct": round(self.avg_demand_pct, 3),
            "avg_inventory_pct": round(self.avg_inventory_pct, 3),
            "avg_risk_ratio": round(self.avg_risk_ratio, 3)
        }