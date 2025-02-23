import pandas as pd
from datetime import datetime

# Create an empty list to store foot traffic data
data = []

while cam.isOpened():
    read, frame = cam.read()
    if not read:
        break

    results = model(frame, conf=0.5)
    persons = sum(1 for result in results[0].boxes if result.cls[0] == 0)

    # Log timestamp and foot traffic count
    data.append({"timestamp": datetime.now(), "count": persons})

    # Annotate frame and display (same as before)
    
# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("foot_traffic.csv", index=False)

