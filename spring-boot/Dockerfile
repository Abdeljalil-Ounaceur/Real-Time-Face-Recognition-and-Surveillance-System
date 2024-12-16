# Build
FROM openjdk:17-jdk-slim AS build

WORKDIR /app

COPY spring-boot/mvnw .
COPY spring-boot/.mvn .mvn

COPY spring-boot/pom.xml .
COPY spring-boot/src ./src

RUN sed -i 's/\r$//' mvnw && chmod +x mvnw
RUN ./mvnw package -DskipTests

# Runtime
FROM openjdk:17-jdk-slim

WORKDIR /app

COPY --from=build /app/target/RealTimeFaceRecognition-0.0.1-SNAPSHOT.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
