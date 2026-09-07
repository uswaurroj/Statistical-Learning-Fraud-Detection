import math
import random

def generate_transactions(num_records=1000, anomaly_ratio=0.05):
    """Generates synthetic transactional data with simulated fraud anomalies."""
    data = []
    num_anomalies = int(num_records * anomaly_ratio)
    
    # Normal transactions (Amount mean=100, std=25)
    for i in range(num_records - num_anomalies):
        amount = max(5, round(random.gauss(100, 25), 2))
        data.append({"tx_id": f"TX_{i+1:04d}", "amount": amount, "label": 0})
        
    # Anomaly/Fraud transactions (Outliers with high amounts)
    for i in range(num_anomalies):
        amount = round(random.uniform(800, 2500), 2)
        data.append({"tx_id": f"TX_FRAUD_{i+1:03d}", "amount": amount, "label": 1})
        
    random.shuffle(data)
    return data

def statistical_anomaly_detection(transactions, threshold_z=2.5):
    """Detects fraud using Z-Score statistical thresholding."""
    amounts = [t["amount"] for t in transactions]
    mean = sum(amounts) / len(amounts)
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in amounts) / len(amounts))
    
    print(f"--- Statistical Anomaly & Fraud Detection Evaluation ---")
    print(f"Dataset Mean Amount: ${mean:.2f} | Std Dev: ${std_dev:.2f}\n")
    
    detected_anomalies = 0
    true_positives = 0
    
    for t in transactions:
        z_score = (t["amount"] - mean) / std_dev
        if abs(z_score) > threshold_z:
            detected_anomalies += 1
            if t["label"] == 1:
                true_positives += 1
                
    print(f"Total Transactions Evaluated: {len(transactions)}")
    print(f"Anomalies Flagged by Model: {detected_anomalies}")
    precision = (true_positives / detected_anomalies) * 100 if detected_anomalies else 0
    print(f"Precision Score: {precision:.2f}%")

if __name__ == "__main__":
    data = generate_transactions(num_records=2000, anomaly_ratio=0.04)
    statistical_anomaly_detection(data, threshold_z=2.5)