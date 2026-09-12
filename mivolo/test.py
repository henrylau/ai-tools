from ultralytics import SAM

# Load the SAM 3 model
model = SAM("sam3_n.pt") # 'n' for nano, or 'l' for large

# Run inference with a text prompt
results = model.predict("/Users/henry/Downloads/HAotHIMagAIxa-u.jpeg", bboxes=None, points=None, labels=["person", "laptop", "coffee"])

# View results
results[0].show()