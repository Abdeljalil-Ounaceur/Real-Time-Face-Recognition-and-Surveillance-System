from kafka import KafkaConsumer, KafkaProducer
import json
import base64
from PIL import Image
import io
from model import predict_faces_info_from_image
import numpy as np

# Kafka configuration
bootstrap_servers = ['kafka:9092']
input_topic = 'first-topic'
output_topic = 'second-topic'

# Create Kafka consumer
consumer = KafkaConsumer(input_topic,
                         bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                         max_partition_fetch_bytes=20971520,  # Set max fetch size to 20MB
                         consumer_timeout_ms=1000)  # Set timeout to 1 second

# Create Kafka producer for sending processed images
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                         max_request_size=20971520)  # Set max request size to 20MB

print("Consumer started. Waiting for images...")

try:
    while True:
        try:
            for message in consumer:
                # Decode the message
                encoded_data = message.value
                session_id, base64_image = encoded_data["sessionId"], encoded_data["base64Image"]
                session_id = session_id.strip().strip('"')
                base64_image = base64_image.strip().strip('"')

                # Decode and preprocess the image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
                image = image.convert('RGB')
                image = np.array(image)

                print("Received an image. Processing...")

                # Process the image
                faces_info = predict_faces_info_from_image(image)
                print(faces_info)

                # Send processed image back to Kafka with session ID
                producer.send(output_topic, {"sessionId":session_id, "results":faces_info})
                producer.flush()

                print("Results sent back.")
        except StopIteration:
            # No message received, continue polling
            continue
except KeyboardInterrupt:
    print("\nExiting consumer...")
finally:
    consumer.close()
    producer.close()