# Real-Time Surveillance System (VGG-Face, Apache Spark, Kafka)

A real-time surveillance system designed to perform facial recognition using VGG-Face, integrated with Apache Spark and Kafka for efficient data processing and real-time streaming, providing an intelligent and scalable solution for security and surveillance applications.

## Features

- **Facial Recognition**: Uses VGG-Face to identify individuals from live video streams.
- **Real-Time Processing**: Apache Spark and Kafka enable real-time data streaming and processing.
- **Data Visualization**: Power BI provides interactive dashboards for monitoring and analyzing surveillance data.
- **Scalability**: Designed to handle large volumes of data with distributed processing.
- **Integration**: Seamless integration of machine learning, big data, and visualization tools for a comprehensive surveillance solution.

## Technologies Used

- VGG-Face (Facial Recognition)
- Apache Spark (Big Data Processing)
- Kafka (Data Streaming)
- Python 3.7 (Programming Language)
- TensorFlow / Keras (Deep Learning Frameworks)
- Docker
- Docker Compose
- Python 3.8+

### Setup and Installation in Docker

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Abdeljalil-Ounaceur/Real-Time-Face-Recognition-and-Surveillance-System.git
   cd Real-Time-Face-Recognition-and-Surveillance-System
   ```

2. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   This command will:
   - Build the necessary Docker images for the model and Spring Boot services
   - Start the containers
   - Set up the networking between services

3. **Stopping the Services**
   ```bash
   docker-compose down
   ```

## Setting Up The model and The App in Local Machine
These are two separate things in one place, the model and the spring-boot app.
The only relationship between the model and the spring-boot app is that they can use kafka to communicate, other than that there is no depedency betwee them. so they are two completely distinct projects. Let's follow 3 insanely simple steps to make our project work without problems.


### Step 1: Kafka Topics
Firts, Run zookeeper and kafka and make sure they are correctly running.

Create two topics `first-topic` and `second-topic`:

in windows:
```
[path to your kafka folder]\windows\kafka-topics --bootstrap-server localhost:9092 --create --topic first-topic
[path to your kafka folder]\windows\kafka-topics --bootstrap-server localhost:9092 --create --topic second-topic
```
in linux:
```
[path to your kafka folder]\bin\kafka-topics.sh --bootstrap-server localhost:9092 --create --topic first-topic
[path to your kafka folder]\bin\kafka-topics.sh --bootstrap-server localhost:9092 --create --topic second-topic
```
<br>

### Step 2: Model Setup

Next, Open the `terminal` in the project location.

Let's create a new virtual environment and install the requirements:

**PLEASE MAKE SURE YOU ARE WORKING WITH PYTHON 3.7 EXACTLY! 
ALSO MAKE SURE THAT ```PIP``` IS UPGRADED TO THE LASTEST VERISON!
OTHERWISE THE REQUIREMENTS WILL FAIL TO INSTALL!!!**

in windows:
```
cd model

python -m venv vggFace
vggFace\Scripts\activate

pip install -r requirements.txt
```

in linux:
```
cd model

python -m venv vggFace
source ./vggFace/bin/activate

pip install -r requirements.txt
```

Let's run the model server located in `best_model/server.py`:
```
cd best_model
py server.py
```

*Optionally, you can go to the `_model_tester/image_sender.py` to test the model:*
```
cd ../_model_tester
py image_sender.py
```
*And try to send an image and see the result appear in the window.*

<br>

### Step 3: Spring Boot Setup
>Note: You need to  have at least Java 17 in JAVA_HOME instead of Java 1.8 because the spring-boot app is based on Java 17.  
>You Also need to install **Maven** if not already installed.

Next, Open another terminal and head to the project location again. and run these commands.
```
cd spring-boot
mvn dependency:resolve
./mvnw spring-boot:run
```
### Step 4: Enjoy Your App

Finally, open the brower and type `localhost:8080`  
**ENJOY!**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
