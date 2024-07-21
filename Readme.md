# Setting Up The model and The App
These are two separate things in one place, the model and the spring-boot app.
The only relationship between the model and the spring-boot app is that they can use kafka to communicate, other than that there is no depedency betwee them. so they are two completely distinct projects. Let's follow 3 insanely simple steps to make our project work without problems.


## Step 1: Kafka Topics
Firts, Run kafka and Create two topics `first-topic` and `second-topic`.
```
cd c:\kafka\bin\windows

.\zookeeper-server-start.bat c:\kafka\config\zookeeper.properties
.\kafka-server-start.bat c:\kafka\config\server.properties

.\kafka-topics localhost:9092 --create --topic first-topic
.\kafka-topics localhost:9092 --create --topic second-topic
```
<br>

## Step 2: Model Setup

Next, Open the `terminal` in the project location.

Let's create a new virtual environment and install the requirements:
```
cd model

py -m venv vggFace
vggFace\Scripts\activate

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

## Step 3: Spring Boot Setup
>Note: You need to  have at least Java 17 in JAVA_HOME instead of Java 1.8 because the spring-boot app is based on Java 17.  
>You Also need to install **Maven** if not already installed, thank you.

Next, Open another terminal and head to the project location again. and run these commands.
```
cd spring-boot
mvn dependency:resolve
./mvnw spring-boot:run
```
## Step 4: Enjoy Your App

Finally, open the brower and type `localhost:8080`  
**ENJOY!**


